# Black Box web app

React (TanStack Start + Vite) frontend for Black Box, wired to the FastAPI backend in `../api/main.py`.

```sh
# from the repository root: starts the API and this app together
./scripts/run_web.sh            # http://127.0.0.1:3000

# or just the frontend (expects the API on http://127.0.0.1:8000; override with BLACKBOX_API_URL)
npm install --ignore-scripts
npx vite dev
```

Checks: `npx tsc --noEmit && npx eslint src && npx vitest run`.

---

# Image Interpreter

refer to the prompt, described images are attached

This project was built with [Lovable](https://lovable.dev).

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/b23c0b96-b772-49e8-8e8c-1311a803b866).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
