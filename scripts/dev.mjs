import { spawn } from "node:child_process";

const processes = [];

function start(command, argumentsList) {
  const child = spawn(command, argumentsList, {
    cwd: process.cwd(),
    env: process.env,
    stdio: "inherit",
  });
  processes.push(child);
  child.on("exit", (code, signal) => {
    if (code && code !== 0) {
      console.error(`${command} exited with code ${code}${signal ? ` (${signal})` : ""}.`);
      stop(code);
    }
  });
  return child;
}

process.env.MPLCONFIGDIR ??= "/tmp/amad-nrp-matplotlib";

function stop(exitCode = 0) {
  for (const child of processes) {
    if (!child.killed) child.kill("SIGTERM");
  }
  setTimeout(() => process.exit(exitCode), 100).unref();
}

process.on("SIGINT", () => stop(0));
process.on("SIGTERM", () => stop(0));

start("uv", ["run", "python", "scripts/dashboard_api.py"]);
start("npm", ["run", "dev:web"]);
