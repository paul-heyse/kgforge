import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import AgentCatalogClient, {
  loadCatalogFromFile,
  searchCatalog,
  type AgentCatalog
} from "../index.js";

const FIXTURE = resolve(
  process.cwd(),
  "..",
  "..",
  "tests",
  "fixtures",
  "agent",
  "catalog_sample.json"
);

async function loadFixture(): Promise<AgentCatalog> {
  return loadCatalogFromFile(FIXTURE);
}

describe("AgentCatalog TypeScript client", () => {
  it("lists packages and modules", async () => {
    const catalog = await loadFixture();
    const client = new AgentCatalogClient(catalog, { repoRoot: process.cwd() });
    expect(client.listPackages().map((pkg) => pkg.name)).toContain("demo");
    expect(client.listModules("demo")[0].qualified).toBe("demo.module");
  });

  it("finds callers and callees", async () => {
    const catalog = await loadFixture();
    const client = new AgentCatalogClient(catalog, { repoRoot: process.cwd() });
    const symbol = catalog.packages[0].modules[0].symbols[0];
    expect(client.findCallers(symbol.symbol_id)).toEqual([]);
    expect(client.findCallees(symbol.symbol_id)).toEqual(["demo.utils.helper"]);
  });

  it("executes lexical search with facets", async () => {
    const catalog = await loadFixture();
    const results = searchCatalog(catalog, "demo utils", { facets: { package: "demo" } });
    expect(results[0].qname).toBe("demo.module.fn");
  });

  it("renders anchor links", async () => {
    const catalog = await loadFixture();
    const client = new AgentCatalogClient(catalog, { repoRoot: "." });
    const symbol = catalog.packages[0].modules[0].symbols[0];
    const anchors = client.openAnchor(symbol.symbol_id);
    expect(anchors.editor).toContain("vscode://file");
    expect(anchors.github).toContain("https://github.com");
  });
});
