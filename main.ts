// free-model-aggregator-02 entry: one server, two wires, one fixed plan.
//
// DESIGN — the fixed 20-call plan (the whole point of this entry):
//   The trial sends exactly six prompts. This server pre-commits, from the
//   brief's manifest alone, a fixed number of upstream calls per prompt:
//   3 calls for a weight-1 prompt and 4 for a weight-2 prompt, 20 in total
//   (2 x 4 + 4 x 3). Every planned call is always issued, in parallel, and
//   there are ZERO retries anywhere: no retry loop, no second attempt after
//   any failure, no scheduled re-dial. A model that answers with a 4xx/5xx
//   (or cannot be reached at all) is marked SPENT and is dropped from every
//   later prompt's plan for the rest of the trial; the later prompt takes the
//   next-best eligible model instead, so its call count stays fixed and the
//   trial total stays 20 as long as enough eligible models remain.
//
//   Why parallel instead of sequential failover: the env waits at most 30 s
//   per prompt, and a sequential chain of slow free models would blow that
//   budget. Why no early cancel: cancelling a scheduled call the moment
//   another provider answers makes the request set depend on arrival timing,
//   and the final's replay must make exactly the live pass's requests. So
//   every planned call goes out, and the answer picked is a pure function of
//   the brief, the request and the gateway's answers: among the successful
//   calls, prefer one whose text carries the prompt's token, then the
//   highest-rated model, then the earliest in the plan. That is deterministic
//   on live and on replay.
//
//   Every upstream call is a non-streaming JSON Chat Completions call, even
//   when the client asked for SSE: the client's stream is synthesised from
//   the chosen answer (one content chunk, one finish chunk with usage, then
//   [DONE] on the OpenAI wire; message_start / content_block_* /
//   message_delta / message_stop on the Anthropic wire). The env checks the
//   stream's shape, never whether the upstream call itself streamed.
//
//   The model field always names the model really dialled (the gateway id,
//   `openrouter/...:free`), on every answer and every streamed chunk.
//
// Determinism: no wall clock, no unseeded randomness. Every choice derives
// from the brief, the requests and the gateway's answers.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

type Json = Record<string, unknown>;
const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);
const stringOf = (v: unknown): string => (typeof v === 'string' ? v : '');
const numberOrNull = (v: unknown): number | null =>
  typeof v === 'number' && Number.isFinite(v) ? v : null;

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
/** Total upstream calls the trial may make: the fixed plan's budget. */
const TOTAL_CALL_BUDGET = 20;
/** Base calls per prompt; the two weight-2 prompts each get one extra. */
const BASE_CALLS_PER_PROMPT = 3;
/** The client waits 30 s per prompt; leave margin for rendering + reply. */
const UPSTREAM_TIMEOUT_MS = 25000;
const MAX_BODY_BYTES = 8 * 1024 * 1024;
const WALLET = 'B5MuBQ53WHCjHDk8WSSMJCGZSzgVgjJJTmqRWhHgYv6G';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key; the gateway swaps it for the entrant's OpenRouter key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

function log(message: string): void {
  process.stderr.write(`aggregator-02: ${message}\n`);
}

// ---------------------------------------------------------------------------
// The brief: the free model list and the six-prompt manifest.
// ---------------------------------------------------------------------------

interface ModelRow {
  id: string;
  provider: string;
  display_name: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface PromptRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
  error: string | null;
}

/** Every JSON object in `text` that carries a string `schema`, in order. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not a JSON object on its own: look on.
    }
  }
  return out;
}

/** Index of the `}` closing the object that opens at `start`, or -1. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function boolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

function modelOf(v: unknown): ModelRow | null {
  if (!isRecord(v)) return null;
  const raw = stringOf(v.id) !== '' ? stringOf(v.id) : stringOf(v.model);
  if (raw === '') return null;
  const named = stringOf(v.provider) !== '' ? stringOf(v.provider) : stringOf(v.wire);
  const id =
    named === 'openrouter' && !raw.toLowerCase().startsWith('openrouter/')
      ? `openrouter/${raw}`
      : raw;
  return {
    id,
    provider: named === '' ? 'openrouter' : named,
    display_name: stringOf(v.display_name) !== '' ? stringOf(v.display_name) : raw,
    rating_milli: numberOrNull(v.rating_milli) ?? 0,
    context_length: numberOrNull(v.context_length),
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
    free: v.free !== false,
  };
}

function promptOf(v: unknown): PromptRow | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      ...(requires.tools === true ? { tools: true } : {}),
      ...(requires.vision === true ? { vision: true } : {}),
      ...(requires.json === true ? { json: true } : {}),
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context)
        ? { min_context: requires.min_context }
        : {}),
    },
  };
}

/** The brief at `path`: the models document and the prompts document. */
function readBrief(path: string): Brief {
  if (path === '') return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text).filter((d) =>
    stringOf(d.schema).includes('free-model-aggregator-02/'),
  );
  let models: ModelRow[] = [];
  let prompts: PromptRow[] = [];
  for (const doc of docs) {
    for (const key of ['models', 'ratings']) {
      if (Array.isArray(doc[key])) {
        models = (doc[key] as unknown[]).flatMap((v) => {
          const m = modelOf(v);
          return m === null ? [] : [m];
        });
      }
    }
    for (const key of ['prompts', 'manifest']) {
      if (Array.isArray(doc[key])) {
        prompts = (doc[key] as unknown[]).flatMap((v) => {
          const p = promptOf(v);
          return p === null ? [] : [p];
        });
      }
    }
  }
  if (models.length === 0) return { models, prompts, error: 'the brief holds no models document' };
  if (prompts.length === 0) return { models, prompts, error: 'the brief holds no prompts document' };
  return { models, prompts, error: null };
}

// ---------------------------------------------------------------------------
// The catalog: exactly the brief's free models, once each.
// ---------------------------------------------------------------------------

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface Catalog {
  providers: {
    id: string;
    base_url: string;
    key_env: string;
    wire: 'openai';
    models: CatalogModel[];
  }[];
}

function buildCatalog(brief: Brief): Catalog {
  const seen = new Set<string>();
  const models: CatalogModel[] = [];
  for (const model of brief.models) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    models.push({ id: model.id, free: true, context_length: model.context_length });
  }
  if (models.length === 0) return { providers: [] };
  return {
    providers: [
      {
        id: 'openrouter',
        // The provider's public endpoint, never the gateway's trial address:
        // the catalog is part of the state export a replay must reproduce.
        base_url: 'https://openrouter.ai/api/v1',
        key_env: 'OPENROUTER_API_KEY',
        wire: 'openai',
        models,
      },
    ],
  };
}

// ---------------------------------------------------------------------------
// The fixed 20-call plan.
// ---------------------------------------------------------------------------

/** Whether a model's capabilities meet one prompt's requirements. */
function meetsRequirements(requires: Requirement, model: ModelRow): boolean {
  if (requires.tools === true && model.tools !== true) return false;
  if (requires.vision === true && model.vision !== true) return false;
  if (requires.json === true && model.json !== true) return false;
  if (
    requires.min_context !== undefined &&
    (model.context_length === null || model.context_length < requires.min_context)
  )
    return false;
  return true;
}

/**
 * Calls per prompt, fixed at startup from the manifest: 3 each, plus one
 * extra for the highest-weight prompts until the total is exactly 20.
 * For the standard manifest (two weight-2, four weight-1) that is 4, 4, 3,
 * 3, 3, 3 in manifest order.
 */
function planSlots(prompts: readonly PromptRow[]): number[] {
  const slots = prompts.map(() => BASE_CALLS_PER_PROMPT);
  let total = slots.reduce((a, b) => a + b, 0);
  const order = prompts
    .map((p, i) => ({ weight: p.weight, i }))
    .sort((a, b) => b.weight - a.weight || a.i - b.i);
  let at = 0;
  while (total < TOTAL_CALL_BUDGET && order.length > 0) {
    slots[order[at % order.length].i] += 1;
    total += 1;
    at += 1;
  }
  return slots;
}

const brief: Brief = readBrief(PROMPT_FILE);
const catalog: Catalog = buildCatalog(brief);
const slots: number[] = planSlots(brief.prompts);
/** Models that have refused once: dropped for the rest of the trial. */
const spentModels = new Set<string>();
/** Which manifest row the next arriving prompt is (the env sends them in order). */
let manifestIndex = 0;

/** The fixed plan for one prompt: the best `count` eligible models, by rating. */
function planFor(prompt: PromptRow, count: number): ModelRow[] {
  const eligible = brief.models
    .filter((m) => m.free && meetsRequirements(prompt.requires, m) && !spentModels.has(m.id))
    .sort((a, b) => b.rating_milli - a.rating_milli || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  return eligible.slice(0, count);
}

// ---------------------------------------------------------------------------
// The gateway: one non-streaming Chat Completions call per planned model.
// ---------------------------------------------------------------------------

function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1')
    ? `${GATEWAY_URL}/chat/completions`
    : `${GATEWAY_URL}/v1/chat/completions`;
}

interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

interface DialResult {
  model: ModelRow;
  ok: boolean;
  answer: Json | null;
  failure: FailureRow | null;
}

/** The error code of a refused gateway answer, as a short string. */
function errorCodeOf(status: number, text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error)) {
      const err = value.error;
      if (typeof err.code === 'string' && err.code !== '') return err.code;
      if (typeof err.code === 'number') return String(err.code);
      if (typeof err.type === 'string' && err.type !== '') return err.type;
    }
    if (isRecord(value) && typeof value.code === 'string' && value.code !== '') return value.code;
  } catch {
    // Not JSON: fall through to the HTTP status.
  }
  return `http_${status}`;
}

/** One planned call. Never retried, never re-dialled: one fetch, one outcome. */
async function dial(model: ModelRow, body: Json): Promise<DialResult> {
  let res: Response;
  try {
    res = await fetch(chatCompletionsUrl(), {
      method: 'POST',
      headers: { 'content-type': 'application/json', authorization: `Bearer ${GATEWAY_KEY}` },
      body: JSON.stringify({ ...body, model: model.id, stream: false }),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
  } catch (err) {
    spentModels.add(model.id);
    return {
      model,
      ok: false,
      answer: null,
      failure: { status: 599, code: 'fetch_error', model: model.id },
    };
  }
  if (!res.ok) {
    spentModels.add(model.id);
    const text = await res.text().catch(() => '');
    return {
      model,
      ok: false,
      answer: null,
      failure: { status: res.status, code: errorCodeOf(res.status, text), model: model.id },
    };
  }
  let answer: unknown;
  try {
    answer = await res.json();
  } catch {
    spentModels.add(model.id);
    return {
      model,
      ok: false,
      answer: null,
      failure: { status: res.status, code: 'bad_json', model: model.id },
    };
  }
  if (!isRecord(answer)) {
    spentModels.add(model.id);
    return {
      model,
      ok: false,
      answer: null,
      failure: { status: res.status, code: 'bad_json', model: model.id },
    };
  }
  return { model, ok: true, answer, failure: null };
}

/** The assistant text of a Chat Completions answer (the env's content check reads this). */
function answerText(answer: Json): string {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  if (typeof message.content === 'string') return message.content;
  if (Array.isArray(message.content)) {
    return message.content
      .flatMap((b) =>
        isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [],
      )
      .join('');
  }
  return '';
}

/** The first `WORD-1234` token in the request's text, or null. */
function tokenOf(body: Json): string | null {
  const texts: string[] = [];
  const collect = (v: unknown): void => {
    if (typeof v === 'string') texts.push(v);
    else if (Array.isArray(v)) v.forEach(collect);
    else if (isRecord(v)) Object.values(v).forEach(collect);
  };
  collect(body.messages ?? body.system);
  for (const text of texts) {
    const match = /[A-Z][A-Z0-9]*-\d{3,}/.exec(text);
    if (match !== null) return match[0];
  }
  return null;
}

/**
 * Pick the answer to serve: among successful calls, prefer one whose text
 * carries the prompt's token, then the highest-rated model, then the
 * earliest in the plan. Pure function of the gateway's answers.
 */
function pickAnswer(results: readonly DialResult[], token: string | null): DialResult | null {
  const ok = results.filter((r) => r.ok && r.answer !== null);
  if (ok.length === 0) return null;
  const withToken =
    token === null ? [] : ok.filter((r) => answerText(r.answer as Json).includes(token));
  const pool = withToken.length > 0 ? withToken : ok;
  const ranked = [...pool].sort(
    (a, b) =>
      b.model.rating_milli - a.model.rating_milli ||
      results.indexOf(a) - results.indexOf(b),
  );
  return ranked[0];
}

// ---------------------------------------------------------------------------
// The wires: Anthropic <-> Chat Completions translation, model field, SSE.
// ---------------------------------------------------------------------------

/** The text of a string or of a block array's text blocks. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** One Anthropic content block as an OpenAI content part; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string')
    return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string')
      return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/**
 * An Anthropic Messages request as a Chat Completions body: system prompt
 * becomes a system message, image blocks become image_url data URLs, tool
 * definitions and tool choice are rewritten, caps and temperature pass
 * through. `stream` is never forwarded: upstream calls are always JSON.
 */
function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0)
    out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema)
              ? raw.input_schema
              : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string')
      out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  return out;
}

/**
 * An OpenAI Chat Completions request, cleaned for the gateway: the dialled
 * model, the cap the gateway requires, no streaming flags, no client-only
 * fields. A missing cap gets 512; a cap over the gateway's 8192 is clamped.
 */
function openAIToUpstream(body: Json, model: string): Json {
  const out: Json = { model, messages: body.messages };
  const cap = typeof body.max_tokens === 'number' ? body.max_tokens : body.max_completion_tokens;
  out.max_tokens = typeof cap === 'number' ? Math.max(1, Math.min(8192, Math.floor(cap))) : 512;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (body.stop !== undefined) out.stop = body.stop;
  if (Array.isArray(body.tools)) out.tools = body.tools;
  if (body.tool_choice !== undefined) out.tool_choice = body.tool_choice;
  if (isRecord(body.response_format)) out.response_format = body.response_format;
  return out;
}

/** The usage of a Chat Completions answer, with a floor of 1 output token. */
function usageOf(answer: Json): { prompt_tokens: number; completion_tokens: number } {
  const usage = isRecord(answer.usage) ? answer.usage : {};
  const input = numberOrZero(usage.prompt_tokens);
  let output = numberOrZero(usage.completion_tokens);
  if (output < 1) output = 1; // A served answer produced at least one token.
  return { prompt_tokens: input, completion_tokens: output };
}

function choiceOf(answer: Json): Json {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  return isRecord(choices[0]) ? choices[0] : {};
}

function messageOf(answer: Json): Json {
  const choice = choiceOf(answer);
  return isRecord(choice.message) ? choice.message : {};
}

function finishOf(answer: Json): string {
  const finish = choiceOf(answer).finish_reason;
  return typeof finish === 'string' && finish !== '' ? finish : 'stop';
}

function toolCallsOf(answer: Json): Json[] {
  const message = messageOf(answer);
  return Array.isArray(message.tool_calls)
    ? message.tool_calls.filter((c): c is Json => isRecord(c))
    : [];
}

/** A Chat Completions answer with `model` set to the id really dialled. */
function openAIJsonWithModel(answer: Json, model: string): Json {
  return { ...answer, model, usage: usageOf(answer) };
}

/** A Chat Completions answer as an Anthropic Messages object. */
function openAIToAnthropic(answer: Json, model: string): Json {
  const content: Json[] = [];
  const text = answerText(answer);
  if (text !== '') content.push({ type: 'text', text });
  for (const call of toolCallsOf(answer)) {
    const fn = isRecord(call.function) ? call.function : {};
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' && call.id !== '' ? call.id : 'toolu_unknown',
      name: typeof fn.name === 'string' ? fn.name : '',
      input: argumentsOf(fn.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = usageOf(answer);
  return {
    id: `msg_${typeof answer.id === 'string' ? answer.id.replace(/^chatcmpl-/, '') : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(finishOf(answer)),
    stop_sequence: null,
    usage: { input_tokens: usage.prompt_tokens, output_tokens: usage.completion_tokens },
  };
}

/** One server-sent event line: `event: <name>` when named, then data, then blank. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split text into chunks of at most `size` characters (streamed output). */
function splitText(text: string, size: number): string[] {
  if (text === '') return [];
  const out: string[] = [];
  for (let i = 0; i < text.length; i += size) out.push(text.slice(i, i + size));
  return out;
}

/**
 * A chosen answer as an OpenAI SSE stream: content deltas, tool-call deltas,
 * exactly one finish chunk carrying the usage, then [DONE]. The model on
 * every chunk is the id really dialled.
 */
function openAIStreamOf(answer: Json, model: string): string {
  const id = typeof answer.id === 'string' ? answer.id : 'chatcmpl-aggregate';
  const created = typeof answer.created === 'number' ? answer.created : 0;
  const usage = usageOf(answer);
  const finish = finishOf(answer);
  const events: string[] = [];
  const text = answerText(answer);
  const chunks = splitText(text, 64);
  const calls = toolCallsOf(answer);
  let first = true;
  for (const part of chunks) {
    events.push(
      sseEvent(
        null,
        {
          id,
          object: 'chat.completion.chunk',
          created,
          model,
          choices: [
            {
              index: 0,
              delta: { ...(first ? { role: 'assistant' } : {}), content: part },
              finish_reason: null,
            },
          ],
        },
      ),
    );
    first = false;
  }
  calls.forEach((call, index) => {
    const fn = isRecord(call.function) ? call.function : {};
    events.push(
      sseEvent(null, {
        id,
        object: 'chat.completion.chunk',
        created,
        model,
        choices: [
          {
            index: 0,
            delta: {
              ...(first ? { role: 'assistant' } : {}),
              tool_calls: [
                {
                  index,
                  id: typeof call.id === 'string' ? call.id : `call_${index}`,
                  type: 'function',
                  function: {
                    name: typeof fn.name === 'string' ? fn.name : '',
                    arguments: typeof fn.arguments === 'string' ? fn.arguments : '',
                  },
                },
              ],
            },
            finish_reason: null,
          },
        ],
      }),
    );
    first = false;
  });
  events.push(
    sseEvent(null, {
      id,
      object: 'chat.completion.chunk',
      created,
      model,
      choices: [{ index: 0, delta: {}, finish_reason: finish }],
      usage,
    }),
  );
  return `${events.join('')}data: [DONE]\n\n`;
}

/**
 * A chosen answer as an Anthropic SSE stream: message_start, one content
 * block per text/tool call (started, delta'd, stopped), then message_delta
 * with the stop reason and output tokens, then message_stop.
 */
function anthropicStreamOf(answer: Json, model: string): string {
  const usage = usageOf(answer);
  const events: string[] = [];
  events.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_stream',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: usage.prompt_tokens, output_tokens: 0 },
      },
    }),
  );
  let index = 0;
  const text = answerText(answer);
  if (text !== '') {
    events.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'text', text: '' },
      }),
    );
    for (const part of splitText(text, 64)) {
      events.push(
        sseEvent('content_block_delta', {
          type: 'content_block_delta',
          index,
          delta: { type: 'text_delta', text: part },
        }),
      );
    }
    events.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  for (const call of toolCallsOf(answer)) {
    const fn = isRecord(call.function) ? call.function : {};
    events.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: {
          type: 'tool_use',
          id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
          name: typeof fn.name === 'string' ? fn.name : '',
          input: {},
        },
      }),
    );
    const args = typeof fn.arguments === 'string' ? fn.arguments : '';
    if (args !== '') {
      events.push(
        sseEvent('content_block_delta', {
          type: 'content_block_delta',
          index,
          delta: { type: 'input_json_delta', partial_json: args },
        }),
      );
    }
    events.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  events.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: stopReasonOf(finishOf(answer)), stop_sequence: null },
      usage: { output_tokens: usage.completion_tokens },
    }),
  );
  events.push(sseEvent('message_stop', { type: 'message_stop' }));
  return events.join('');
}

// ---------------------------------------------------------------------------
// HTTP plumbing.
// ---------------------------------------------------------------------------

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders?: Json): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    'content-length': String(Buffer.byteLength(text)),
  };
  if (extraHeaders !== undefined) {
    for (const [k, v] of Object.entries(extraHeaders)) {
      if (typeof v === 'string') headers[k] = v;
    }
  }
  res.writeHead(status, headers);
  res.end(text);
}

function sendText(res: ServerResponse, status: number, text: string, extraHeaders?: Json): void {
  const headers: Record<string, string> = {
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
    'content-length': String(Buffer.byteLength(text)),
  };
  if (extraHeaders !== undefined) {
    for (const [k, v] of Object.entries(extraHeaders)) {
      if (typeof v === 'string') headers[k] = v;
    }
  }
  res.writeHead(status, headers);
  res.end(text);
}

function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

function openAIError(message: string, type: string): Json {
  return { error: { message, type, code: null } };
}

function sendError(
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  status: number,
  type: string,
  message: string,
  extraHeaders?: Json,
): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), extraHeaders);
  else sendJson(res, status, openAIError(message, type), extraHeaders);
}

/** GET /v1/models, in the shape of the auth header it got. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = brief.models
    .filter((m) => m.free)
    .map((m) => ({ id: m.id, provider: m.provider, display_name: m.display_name }));
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, {
      data: list.map((m) => ({
        type: 'model',
        id: m.id,
        display_name: m.display_name,
        created_at: '2026-01-01T00:00:00Z',
      })),
      has_more: false,
    });
  } else {
    sendJson(res, 200, {
      object: 'list',
      data: list.map((m) => ({
        id: m.id,
        object: 'model',
        created: 1767225600,
        owned_by: m.provider,
      })),
    });
  }
}

/**
 * Serve one prompt: run its fixed plan (every planned call, in parallel,
 * zero retries), pick the best successful answer, and render it in the
 * wire the client used. Failures are reported in the
 * x-makefun-upstream-failures header so the failover column can see them.
 */
async function serve(
  req: IncomingMessage,
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    sendError(res, wire, 400, 'invalid_request_error', 'the request body is not one JSON object');
    return;
  }
  const row = brief.prompts[manifestIndex];
  manifestIndex += 1;
  const count = slots[manifestIndex - 1] ?? BASE_CALLS_PER_PROMPT;
  const prompt: PromptRow =
    row ?? {
      id: `extra-${manifestIndex}`,
      api: wire,
      stream: body.stream === true,
      weight: 1,
      requires: {
        ...(Array.isArray(body.tools) && body.tools.length > 0 ? { tools: true } : {}),
        ...(isRecord(body.response_format) ? { json: true } : {}),
      },
    };
  const plan = planFor(prompt, count);
  if (plan.length === 0) {
    sendError(res, wire, 503, 'no_free_model', 'no free model in the brief meets this request');
    return;
  }
  const upstreamBody =
    wire === 'anthropic' ? anthropicToOpenAI(body, plan[0].id) : openAIToUpstream(body, plan[0].id);
  // The upstream body is rebuilt per model (only the `model` field differs),
  // so the plan's calls are identical apart from the model they name.
  const results = await Promise.allSettled(plan.map((model) => dial(model, upstreamBody)));
  const settled: DialResult[] = results.map((r, i) =>
    r.status === 'fulfilled'
      ? r.value
      : {
          model: plan[i],
          ok: false,
          answer: null,
          failure: { status: 599, code: 'fetch_error', model: plan[i].id },
        },
  );
  // A rejection inside Promise.allSettled cannot happen (dial never throws),
  // but a settled rejection still spends the model: drop it like any failure.
  for (const r of settled) if (!r.ok) spentModels.add(r.model.id);
  const failures = settled
    .filter((r) => r.failure !== null)
    .map((r) => r.failure as FailureRow);
  const headers: Json =
    failures.length > 0
      ? { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, 8)) }
      : {};
  const token = tokenOf(body);
  const picked = pickAnswer(settled, token);
  if (picked === null || picked.answer === null) {
    sendError(
      res,
      wire,
      502,
      wire === 'anthropic' ? 'api_error' : 'upstream_error',
      'every planned model refused this prompt',
      headers,
    );
    return;
  }
  const modelId = picked.model.id;
  const answer = picked.answer;
  log(`served ${prompt.id} with ${modelId} (${settled.filter((r) => r.ok).length}/${settled.length} planned calls answered)`);
  if (wire === 'openai') {
    if (body.stream === true) sendText(res, 200, openAIStreamOf(answer, modelId), headers);
    else sendJson(res, 200, openAIJsonWithModel(answer, modelId), headers);
  } else {
    if (body.stream === true) sendText(res, 200, anthropicStreamOf(answer, modelId), headers);
    else sendJson(res, 200, openAIToAnthropic(answer, modelId), headers);
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve(req, res, 'anthropic');
    return;
  }
  sendError(
    res,
    url.pathname === '/v1/messages' ? 'anthropic' : 'openai',
    404,
    'not_found',
    `no route ${req.method ?? ''} ${url.pathname}`,
  );
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; retry briefly if the env is not up yet. */
async function register(port: number): Promise<void> {
  const payload = JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog });
  let lastError: unknown = null;
  for (let attempt = 0; attempt < 20; attempt += 1) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: payload,
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') {
        log(`registered: the trial holds ${reply.requests} prompts; plan slots ${slots.join('+')}=${slots.reduce((a, b) => a + b, 0)}`);
      }
      return;
    } catch (err) {
      lastError = err;
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }
  throw new Error(`register failed: ${(lastError as Error)?.message ?? String(lastError)}`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json();
          if (isRecord(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(
          res,
          req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai',
          500,
          'internal_error',
          'the server failed to answer',
        );
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout');
  log(`listening on 0.0.0.0:${port} with ${brief.models.filter((m) => m.free).length} free models`);
  await register(port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
