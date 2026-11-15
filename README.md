[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/adenadji06/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/adenadji06/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

# free-model-aggregator-02 entry

One server (`main.ts`, Node 22 stdlib only) that serves the OpenAI Chat
Completions API and the Anthropic Messages API — JSON and SSE — over the
fourteen free OpenRouter models a trial's brief lists, dialling them through
the grader's gateway.

Bounty: https://make.fun/c/free-model-aggregator-02

## The fixed 20-call plan

The trial sends exactly six prompts, and this server pre-commits a fixed
number of upstream calls per prompt from the brief's manifest alone: 3 calls
for a weight-1 prompt, 4 for a weight-2 prompt — 20 calls in total. That
number is the whole design:

- **Zero retries.** Every planned call is issued once, in parallel. There is
  no retry loop, no second attempt after any failure, no scheduled re-dial.
- **Drop on first failure.** A model that answers 4xx/5xx (or cannot be
  reached) is marked spent and dropped from every later prompt's plan; the
  later prompt takes the next-best eligible model instead, so its call count
  stays fixed and the trial total stays 20.
- **No early cancel.** Every planned call always goes out, so the request set
  never depends on arrival timing and the final's replay makes exactly the
  live pass's requests.
- **Best successful answer wins.** Among the successful calls, prefer one
  whose text carries the prompt's token, then the highest-rated model
  (Inkling 2 Small at 1475 for plain prompts, Nemotron 3.5 Super VL at 1303
  for JSON-mode prompts), then the earliest in the plan.

Every upstream call is a non-streaming JSON Chat Completions call, even when
the client asked for SSE — the client's stream is synthesised from the chosen
answer. Failures are reported in the `x-makefun-upstream-failures` header so
the failover column can see them.

## Files

- `main.ts` — the server: brief parsing, catalog, fixed plan, both wires,
  register, work poll. No dependencies beyond the Node 22 standard library.
- `LICENSE` — MIT.
- `.makefun/owner.json` — the payout wallet claim.

## Run

```sh
node main.ts --list   # prints the catalog document, byte for byte as registered
node main.ts          # serves /v1/chat/completions, /v1/messages, /v1/models
```

Deterministic: no wall clock, no unseeded randomness — every choice derives
from the brief, the requests and the gateway's answers, so a replay leaves
the same state export.
