# aiChat

## Overview
openaiChat is a lightweight C web server that lets you stage round-table conversations between multiple Open WebUI-managed
models. Launch the binary, open the served page in your browser, and configure up to six companions—each with a friendly
name and a backing Open WebUI model tag. The application coordinates turn-based conversations through the Open WebUI HTTP API and
streams every reply back to the UI as soon as it is generated.

### Key capabilities
* Streams conversation updates message-by-message so you can follow the debate in real time.
* Discovers the available Open WebUI models at start-up and highlights if a previously selected model disappears.
* Provides a built-in HTML interface served from `/`—no external assets or tooling required.
* Exposes JSON endpoints so other tools can fetch the Open WebUI model list or trigger conversations programmatically.

## Prerequisites
* A running Open WebUI instance reachable from the machine that launches aiChat. The default endpoint is
  `http://127.0.0.1:8080/ollama/api/generate`.
* Build tools: `gcc`, `make`, and `pkg-config`.
* Development headers for `libcurl` and `json-c`.

Before building, run the provided `./configure` script to confirm that the required toolchain and libraries are
discoverable. The script prints specific installation hints for anything that is missing.

On Debian/Ubuntu systems you can install the dependencies with:

```
sudo apt-get update
sudo apt-get install -y \
  build-essential pkg-config \
  libcurl4-openssl-dev \
  libjson-c-dev
```

## Building

### Linux (GNU Make)
1. `./configure`
2. Update `config.h` with your Open WebUI API key.
3. `make`
4. Run the resulting `./aichat` binary.

### Windows (MinGW / MSYS2 Make)
1. Launch an MSYS2 or MinGW shell that provides the GNU toolchain and `pkg-config`.
2. `./configure`
3. `make -f Makefile.win`
4. Run the generated `aichat.exe` from the same shell.

## Running the server
* Before launching the server, provide your Open WebUI API key by either exporting `WEBUI_API_KEY` in the environment or
  by editing `config.h` and rebuilding. The bundled placeholder is ignored automatically, so leaving it untouched simply
  means aiChat will make unauthenticated requests.
* Execute `./openaichat` after building. On success the server prints the URL it bound to (defaults to
  `http://127.0.0.1:4000`).
* If the preferred port is taken, aiChat retries up to three higher ports before giving up.
* Override the listening port by exporting `AICHAT_PORT`, e.g. `AICHAT_PORT=19000 ./openaichat`.
* Point aiChat at a different Open WebUI deployment by setting `OLLAMA_URL` to the proxied `/ollama/api/generate` endpoint.
  For example, if the Open WebUI interface lives at `http://127.0.0.1:8080/`, export `OLLAMA_URL=http://127.0.0.1:8080/ollama/api/generate`.
* aiChat prints the derived model discovery URLs at startup. If model loading fails, copy the suggested `curl` command from the
  log or the UI diagnostics panel and run it from the same machine to confirm connectivity.
* Stop the server with <kbd>Ctrl</kbd>+<kbd>C</kbd> in the terminal where it is running.

## Using the web UI
1. Browse to the printed URL after starting the server.
2. Enter a topic and choose how many turns to run (the value is clamped between 1 and 12).
3. Configure the participant roster:
   * Two example companions—Astra (`gemma:2b`) and Nova (`llama3:8b`)—are pre-populated.
   * Use **Add participant** to introduce more companions (up to a total of six). Remove an entry with the **Remove**
     button next to it.
   * Pick a model for each participant. aiChat fetches the list from `/models`; if a desired model disappears, the UI
     raises a warning and keeps the selection for when it returns.
4. Press **Start conversation** to begin. Status messages above the form show whether aiChat is waiting on models, talking
   to Open WebUI, or complete, and the API indicator reports the connection state.
5. Watch the transcript panel fill in as each response arrives. Messages show the speaker’s friendly name, chosen model,
   and reply text.

## Troubleshooting Open WebUI connectivity

If the status indicator reports “Unable to load models from Open WebUI”, run through the checklist below:

1. Expand the **API troubleshooting tips** panel in the UI. It calls out the exact URLs aiChat is using and offers a ready-made
   `curl` command (including the `Authorization` header when applicable).
2. Run the command from the machine hosting aiChat. A successful response should echo the Open WebUI model catalogue.
3. Review the terminal where aiChat is running. Failed model discovery attempts now print the HTTP status or CURL error code
   along with reminder commands you can retry manually.
4. Adjust `OLLAMA_URL` if needed so that it points at your deployment’s proxied `/ollama/api/generate` endpoint, then restart
   aiChat. The `/diagnostics` endpoint (served by aiChat itself) also reports the currently configured URLs in JSON form.

## API reference

### `GET /`
Serves the single-page HTML interface described above.

### `GET /models`
Returns the available Open WebUI models in the shape:

```json
{ "models": [ { "name": "LLaMA 3 8B", "model": "llama3:8b" }, ... ] }
```

aiChat queries Open WebUI's `/api/models` endpoint for this list and falls back to the proxied `/ollama/api/tags` endpoint if
needed. If every attempt fails, the server responds with `502 Bad Gateway` and a JSON error message.

### `POST /chat`
Starts a turn-based conversation. The request body must be JSON with the following fields:

```json
{
  "topic": "Space exploration strategies",
  "turns": 3,
  "participants": [
    { "name": "Astra", "model": "gemma:2b" },
    { "name": "Nova", "model": "llama3:8b" }
  ]
}
```

`turns` is clamped between 1 and 12, and aiChat ignores participants without a model. Friendly names default to themed
values (Astra, Nova, Cosmo, etc.) when omitted.

Responses are streamed back as chunked NDJSON events (`application/x-ndjson`). Expect a sequence of objects with the
following `type` values:

* `start` — echo of the topic, turn count, and resolved participant roster.
* `message` — a single participant reply, including `participantIndex`, `name`, `model`, and `text`.
* `complete` — signals the discussion finished successfully.
* `error` — a terminal error message if the conversation could not be completed.

## Roadmap
* Provide a transcript export option (text/JSON) after the session ends.
* Allow saving and reusing favourite participant rosters.
* Offer a command-line mode that prints the transcript without launching the browser UI.

## License
aiChat is distributed under the MIT License. See [LICENSE](LICENSE) for details.
