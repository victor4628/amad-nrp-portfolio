import assert from "node:assert/strict";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the AMAD-NRP dashboard shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(
    html,
    /<title>AMAD-Enhanced Network Risk Parity for S&amp;P 100<\/title>/i,
  );
  assert.doesNotMatch(html, /og-research-dashboard\.png/i);
  assert.match(html, /Run backtest/);
  assert.match(html, /24 months \(default\)/);
  assert.match(html, /12 months/);
  assert.doesNotMatch(html, /36 months/);
  assert.match(html, /64 days/);
  assert.match(html, /126 days \(default\)/);
  assert.doesNotMatch(html, /21 days|42 days|63 days/);
  assert.match(html, /No backtest has been run in this session/);
  assert.doesNotMatch(html, /Latest minimum spanning tree|Loading research snapshot/i);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton|Your site is taking shape/i);
});
