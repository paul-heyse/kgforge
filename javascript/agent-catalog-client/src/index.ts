import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

export interface Anchors {
  start_line?: number | null;
  end_line?: number | null;
  cst_fingerprint?: string | null;
  remap_order?: Array<Record<string, unknown>>;
}

export interface Quality {
  pyright_status?: string | null;
  pyrefly_status?: string | null;
  ruff_rules?: string[];
  pydoclint_parity?: boolean | null;
  docstring_coverage?: number | null;
  doctest_status?: string | null;
}

export interface Metrics {
  complexity?: number | null;
  loc?: number | null;
  last_modified?: string | null;
  codeowners?: string[];
  stability?: string | null;
  deprecated?: boolean | null;
}

export interface AgentHints {
  intent_tags?: string[];
  safe_ops?: string[];
  tests_to_run?: string[];
  perf_budgets?: string[];
  breaking_change_notes?: string[];
}

export interface ChangeImpactTest {
  file: string;
  lines?: number[];
  reason?: string;
  windows?: Array<{ start: number; end: number }>;
}

export interface ChangeImpact {
  callers?: string[];
  callees?: string[];
  tests?: ChangeImpactTest[];
  codeowners?: string[];
  churn_last_n?: number | null;
}

export interface Exemplar {
  title?: string;
  language?: string;
  snippet?: string;
  counter_example?: string;
  negative_prompts?: string[];
  context_notes?: string;
}

export interface SymbolEntry {
  qname: string;
  kind: string;
  symbol_id: string;
  docfacts?: Record<string, unknown> | null;
  anchors: Anchors;
  quality: Quality;
  metrics: Metrics;
  agent_hints: AgentHints;
  change_impact: ChangeImpact;
  exemplars?: Exemplar[];
}

export interface ModuleGraph {
  imports?: string[];
  calls?: Array<Record<string, unknown>>;
}

export interface ModuleEntry {
  name: string;
  qualified: string;
  source: Record<string, string | undefined>;
  pages: Record<string, string | undefined>;
  imports?: string[];
  symbols: SymbolEntry[];
  graph?: ModuleGraph;
}

export interface PackageEntry {
  name: string;
  modules: ModuleEntry[];
}

export interface ShardIndexEntry {
  name: string;
  path: string;
  modules?: number;
}

export interface ShardIndex {
  index: string;
  packages: ShardIndexEntry[];
}

export interface SemanticIndexMetadata {
  index: string;
  mapping: string;
  model?: string;
  dimension?: number;
  count?: number;
}

export interface SearchConfiguration {
  alpha?: number;
  candidate_pool?: number;
  lexical_fields?: string[];
}

export interface AgentCatalog {
  version: string;
  generated_at: string;
  repo: Record<string, string>;
  link_policy: Record<string, unknown>;
  artifacts: Record<string, unknown>;
  packages: PackageEntry[];
  shards?: ShardIndex | null;
  semantic_index?: SemanticIndexMetadata | null;
  search?: SearchConfiguration | null;
}

export interface SearchResult {
  symbolId: string;
  score: number;
  lexicalScore: number;
  vectorScore: number;
  package: string;
  module: string;
  qname: string;
}

export interface SearchOptions {
  facets?: Record<string, string>;
  k?: number;
}

interface SearchDocument {
  symbol: SymbolEntry;
  package: string;
  module: string;
  text: string;
}

function normaliseArray<T>(value: T[] | undefined | null): T[] {
  return Array.isArray(value) ? value : [];
}

function normaliseNumber(value: unknown): number | null {
  if (value === null || value === undefined) {
    return null;
  }
  const parsed = Number(value);
  return Number.isNaN(parsed) ? null : parsed;
}

async function loadJson(path: string): Promise<any> {
  const raw = await readFile(path, "utf8");
  return JSON.parse(raw);
}

export async function loadCatalogFromFile(
  path: string,
  options: { loadShards?: boolean } = {}
): Promise<AgentCatalog> {
  const payload = (await loadJson(path)) as AgentCatalog;
  if (
    options.loadShards !== false &&
    (!payload.packages || payload.packages.length === 0) &&
    payload.shards &&
    typeof payload.shards.index === "string"
  ) {
    const root = dirname(path);
    const indexPath = resolve(root, payload.shards.index);
    const indexPayload = (await loadJson(indexPath)) as {
      packages?: ShardIndexEntry[];
    };
    const packages: PackageEntry[] = [];
    const shardEntries = Array.isArray(indexPayload.packages) ? indexPayload.packages : [];
    for (const entry of shardEntries) {
      if (!entry?.path) {
        continue;
      }
      const shardPath = resolve(dirname(indexPath), entry.path);
      const shardPayload = await loadJson(shardPath);
      if (shardPayload) {
        packages.push(shardPayload as PackageEntry);
      }
    }
    payload.packages = packages;
  }
  return payload;
}

function buildDocument(packageName: string, module: ModuleEntry, symbol: SymbolEntry): SearchDocument {
  const hints = normaliseArray(symbol.agent_hints.intent_tags).join(" ");
  const tests = normaliseArray(symbol.agent_hints.tests_to_run).join(" ");
  const docfacts = symbol.docfacts ?? {};
  const summary = typeof docfacts["summary"] === "string" ? docfacts["summary"] : "";
  const docstring = typeof docfacts["docstring"] === "string" ? docfacts["docstring"] : "";
  const text = [
    symbol.qname,
    module.qualified,
    packageName,
    summary,
    docstring,
    hints,
    tests
  ]
    .filter(Boolean)
    .join(" ");
  return { symbol, package: packageName, module: module.qualified, text };
}

function collectDocuments(catalog: AgentCatalog): SearchDocument[] {
  const documents: SearchDocument[] = [];
  for (const pkg of catalog.packages ?? []) {
    for (const module of pkg.modules ?? []) {
      for (const symbol of module.symbols ?? []) {
        documents.push(buildDocument(pkg.name, module, symbol));
      }
    }
  }
  return documents;
}

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9_]+/)
    .filter(Boolean);
}

function lexicalScore(query: string, document: SearchDocument): number {
  const queryTokens = tokenize(query);
  if (queryTokens.length === 0) {
    return 0;
  }
  const docTokens = tokenize(document.text);
  if (docTokens.length === 0) {
    return 0;
  }
  const docCounts = new Map<string, number>();
  for (const token of docTokens) {
    docCounts.set(token, (docCounts.get(token) ?? 0) + 1);
  }
  let score = 0;
  for (const token of queryTokens) {
    score += docCounts.get(token) ?? 0;
  }
  if (document.symbol.qname.toLowerCase().includes(query.toLowerCase())) {
    score += 2;
  }
  return score;
}

function applyFacets(document: SearchDocument, facets?: Record<string, string>): boolean {
  if (!facets) {
    return true;
  }
  const symbol = document.symbol;
  for (const [key, raw] of Object.entries(facets)) {
    const value = raw.toLowerCase();
    if (key === "package" && document.package.toLowerCase() !== value) {
      return false;
    }
    if (key === "module" && document.module.toLowerCase() !== value) {
      return false;
    }
    if (key === "stability") {
      const stability = symbol.metrics.stability?.toLowerCase() ?? "";
      if (stability !== value) {
        return false;
      }
    }
    if (key === "parity") {
      const parity = symbol.quality.pydoclint_parity;
      if (value === "pass" && parity !== true) {
        return false;
      }
      if (value === "fail" && parity !== false) {
        return false;
      }
    }
    if (key === "coverage") {
      const threshold = Number(value.replace(/[^0-9.]/g, ""));
      if (!Number.isNaN(threshold)) {
        const ratio = symbol.quality.docstring_coverage ?? 0;
        const normalized = threshold > 1 ? threshold / 100 : threshold;
        if (ratio < normalized) {
          return false;
        }
      }
    }
    if (key === "churn") {
      const churnValue = normaliseNumber(symbol.change_impact.churn_last_n) ?? 0;
      const target = Number(value.replace(/[^0-9]/g, ""));
      if (!Number.isNaN(target) && churnValue < target) {
        return false;
      }
    }
    if (key === "deprecated") {
      const expected = value === "true" || value === "1";
      if ((symbol.metrics.deprecated ?? false) !== expected) {
        return false;
      }
    }
  }
  return true;
}

export function searchCatalog(
  catalog: AgentCatalog,
  query: string,
  options: SearchOptions = {}
): SearchResult[] {
  const documents = collectDocuments(catalog);
  const filtered = documents.filter((document) => applyFacets(document, options.facets));
  const results = filtered
    .map((document) => {
      const lexical = lexicalScore(query, document);
      return {
        symbolId: document.symbol.symbol_id,
        score: lexical,
        lexicalScore: lexical,
        vectorScore: 0,
        package: document.package,
        module: document.module,
        qname: document.symbol.qname
      } satisfies SearchResult;
    })
    .filter((result) => result.score > 0)
    .sort((a, b) => b.score - a.score);
  const limit = options.k ?? 10;
  return results.slice(0, Math.max(1, limit));
}

export class AgentCatalogClient {
  readonly catalog: AgentCatalog;
  readonly catalogPath?: string;
  readonly repoRoot: string;

  constructor(catalog: AgentCatalog, options: { repoRoot?: string; catalogPath?: string } = {}) {
    this.catalog = catalog;
    this.catalogPath = options.catalogPath;
    this.repoRoot = options.repoRoot ?? process.cwd();
  }

  static async fromFile(
    path: string,
    options: { repoRoot?: string; loadShards?: boolean } = {}
  ): Promise<AgentCatalogClient> {
    const catalog = await loadCatalogFromFile(path, { loadShards: options.loadShards });
    const repoRoot = options.repoRoot ?? dirname(path);
    return new AgentCatalogClient(catalog, { repoRoot, catalogPath: path });
  }

  listPackages(): PackageEntry[] {
    return [...(this.catalog.packages ?? [])];
  }

  listModules(packageName: string): ModuleEntry[] {
    const pkg = this.catalog.packages.find((entry) => entry.name === packageName);
    if (!pkg) {
      throw new Error(`Unknown package: ${packageName}`);
    }
    return [...pkg.modules];
  }

  getModule(qualified: string): ModuleEntry | undefined {
    for (const pkg of this.catalog.packages ?? []) {
      const module = pkg.modules.find((entry) => entry.qualified === qualified);
      if (module) {
        return module;
      }
    }
    return undefined;
  }

  iterSymbols(): SymbolEntry[] {
    const symbols: SymbolEntry[] = [];
    for (const pkg of this.catalog.packages ?? []) {
      for (const module of pkg.modules ?? []) {
        symbols.push(...(module.symbols ?? []));
      }
    }
    return symbols;
  }

  getSymbol(symbolId: string): SymbolEntry | undefined {
    return this.iterSymbols().find((symbol) => symbol.symbol_id === symbolId);
  }

  findCallers(symbolId: string): string[] {
    const symbol = this.getSymbol(symbolId);
    if (!symbol) {
      throw new Error(`Unknown symbol: ${symbolId}`);
    }
    return normaliseArray(symbol.change_impact.callers);
  }

  findCallees(symbolId: string): string[] {
    const symbol = this.getSymbol(symbolId);
    if (!symbol) {
      throw new Error(`Unknown symbol: ${symbolId}`);
    }
    return normaliseArray(symbol.change_impact.callees);
  }

  changeImpact(symbolId: string): ChangeImpact {
    const symbol = this.getSymbol(symbolId);
    if (!symbol) {
      throw new Error(`Unknown symbol: ${symbolId}`);
    }
    return symbol.change_impact;
  }

  suggestTests(symbolId: string): ChangeImpactTest[] {
    return normaliseArray(this.changeImpact(symbolId).tests);
  }

  search(query: string, options: SearchOptions = {}): SearchResult[] {
    return searchCatalog(this.catalog, query, options);
  }

  openAnchor(symbolId: string): Record<string, string> {
    const symbol = this.getSymbol(symbolId);
    if (!symbol) {
      throw new Error(`Unknown symbol: ${symbolId}`);
    }
    const moduleName = symbol.qname.includes(".")
      ? symbol.qname.slice(0, symbol.qname.lastIndexOf("."))
      : symbol.qname;
    const module = this.getModule(moduleName);
    if (!module) {
      throw new Error(`Unknown module for symbol: ${symbolId}`);
    }
    const sourcePath = module.source?.path;
    if (!sourcePath) {
      throw new Error(`Symbol source path missing from catalog for ${symbolId}`);
    }
    const startLine = symbol.anchors.start_line ?? 1;
    const linkPolicy = this.catalog.link_policy ?? {};
    const editorTemplate =
      typeof linkPolicy["editor_template"] === "string"
        ? (linkPolicy["editor_template"] as string)
        : "vscode://file/{path}:{line}";
    const githubTemplate =
      typeof linkPolicy["github_template"] === "string"
        ? (linkPolicy["github_template"] as string)
        : "https://github.com/{org}/{repo}/blob/{sha}/{path}#L{line}";
    const repoRoot = this.repoRoot;
    const absolute = resolve(repoRoot, sourcePath);
    const editorLink = editorTemplate
      .replace("{path}", absolute)
      .replace("{line}", String(startLine));
    const githubVars = (linkPolicy["github"] as Record<string, string | undefined>) ?? {};
    const githubLink = githubTemplate
      .replace("{org}", githubVars.org ?? "")
      .replace("{repo}", githubVars.repo ?? "")
      .replace("{sha}", githubVars.sha ?? "")
      .replace("{path}", sourcePath)
      .replace("{line}", String(startLine));
    return { editor: editorLink, github: githubLink };
  }
}

export default AgentCatalogClient;
