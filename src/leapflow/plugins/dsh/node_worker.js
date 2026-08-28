"use strict";

// Restricted DSH/Cordis worker. stdout is reserved for protocol NDJSON; all
// diagnostics and plugin console output are redirected to stderr.
const path = require("path");
const readline = require("readline");
const { AsyncLocalStorage } = require("async_hooks");
const { pathToFileURL } = require("url");

const VERSION = 1;
const MAX_LINE_BYTES = Number(process.env.LEAPFLOW_DSH_MAX_LINE_BYTES || 1000000);
const CAPABILITY_TIMEOUT_MS = Number(process.env.LEAPFLOW_DSH_CAPABILITY_TIMEOUT_MS || 120000);
const sourceRoot = path.resolve(process.argv[2] || "");
const entryPoint = String(process.argv[4] || "");

const tools = new Map();
const handlers = new Map();
const pendingCapabilities = new Map();
const invocationContext = new AsyncLocalStorage();

function stderrLine(level, args) {
  const rendered = args.map((value) => {
    if (typeof value === "string") return value;
    try { return JSON.stringify(value); } catch (_) { return String(value); }
  }).join(" ");
  process.stderr.write(`[dsh:${level}] ${rendered}\n`);
}

for (const level of ["log", "info", "warn", "error", "debug"]) {
  console[level] = (...args) => stderrLine(level, args);
}

function writeMessage(message) {
  let encoded;
  try {
    encoded = JSON.stringify(message);
  } catch (error) {
    encoded = JSON.stringify({
      version: VERSION,
      type: "response",
      request_id: String(message && message.request_id || "serialization-error"),
      ok: false,
      error: `Result is not JSON serializable: ${String(error && error.message || error)}`,
      error_type: "serialization_error",
    });
  }
  if (Buffer.byteLength(encoded, "utf8") > MAX_LINE_BYTES) {
    encoded = JSON.stringify({
      version: VERSION,
      type: "response",
      request_id: String(message && message.request_id || "oversize"),
      ok: false,
      error: `Worker response exceeds ${MAX_LINE_BYTES} bytes`,
      error_type: "response_too_large",
    });
  }
  process.stdout.write(encoded + "\n");
}

function normalizeSchema(parameters) {
  if (parameters === undefined || parameters === null) {
    return { type: "object", properties: {} };
  }
  if (typeof parameters !== "object" || Array.isArray(parameters)) {
    throw new Error("DSH tool parameters must be an object");
  }
  const looksLikeJsonSchema = Object.prototype.hasOwnProperty.call(parameters, "type")
    || Object.prototype.hasOwnProperty.call(parameters, "properties")
    || Object.prototype.hasOwnProperty.call(parameters, "required")
    || Object.prototype.hasOwnProperty.call(parameters, "additionalProperties");
  if (looksLikeJsonSchema) {
    if (parameters.type !== undefined && parameters.type !== "object") {
      throw new Error("DSH tool parameters JSON schema must have type 'object'");
    }
    const properties = parameters.properties === undefined ? {} : parameters.properties;
    if (!properties || typeof properties !== "object" || Array.isArray(properties)) {
      throw new Error("DSH tool parameters schema.properties must be an object");
    }
    const required = parameters.required === undefined ? [] : parameters.required;
    if (!Array.isArray(required) || required.some((name) => typeof name !== "string")) {
      throw new Error("DSH tool parameters schema.required must be an array of strings");
    }
    if (required.some((name) => !Object.prototype.hasOwnProperty.call(properties, name))) {
      throw new Error("DSH tool parameters schema.required references an unknown property");
    }
    return { ...parameters, type: "object", properties, required };
  }
  const properties = {};
  const required = [];
  for (const [name, spec] of Object.entries(parameters)) {
    if (!spec || typeof spec !== "object" || Array.isArray(spec)) {
      throw new Error(`DSH tool parameter ${name} must be an object`);
    }
    const normalized = { ...spec };
    delete normalized.required;
    properties[name] = normalized;
    if (spec.required === true) required.push(name);
  }
  const schema = { type: "object", properties };
  if (required.length) schema.required = required;
  return schema;
}

function registerTool(_ctx, tool) {
  if (!tool || typeof tool !== "object") throw new Error("registerTool requires a tool object");
  const name = String(tool.name || "");
  if (!/^[a-z][a-z0-9_]*$/.test(name)) {
    throw new Error(`DSH tool name must use lowercase snake_case: ${name}`);
  }
  if (typeof tool.execute !== "function") throw new Error(`DSH tool ${name} has no execute function`);
  if (tools.has(name)) throw new Error(`Duplicate DSH tool: ${name}`);
  tools.set(name, {
    name,
    description: String(tool.description || `DSH tool ${name}`),
    parameters_schema: normalizeSchema(tool.parameters),
    execute: tool.execute,
  });
  return () => tools.delete(name);
}

function capabilityCall(capability, argumentsValue) {
  const parent = invocationContext.getStore();
  if (!parent) return Promise.reject(new Error("Capability requested outside a tool invocation"));
  const requestId = `${parent}:${Date.now()}:${Math.random().toString(16).slice(2)}`;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      pendingCapabilities.delete(requestId);
      reject(new Error(`Capability ${capability} timed out`));
    }, CAPABILITY_TIMEOUT_MS);
    pendingCapabilities.set(requestId, { resolve, reject, timer });
    writeMessage({
      version: VERSION,
      type: "capability_request",
      request_id: requestId,
      parent_request_id: parent,
      capability,
      arguments: argumentsValue || {},
    });
  });
}

const shellService = {
  resolve(spec) { return { ...(spec || {}) }; },
  async run(spec) {
    return capabilityCall("compat.shell.run", {
      command: String(spec && spec.command || ""),
      timeoutMs: Number(spec && spec.timeoutMs || 20000),
      stdoutMaxBytes: Number(spec && spec.stdoutMaxBytes || 1048576),
    });
  },
};

const context = {
  get(name) {
    if (name === "shell") return shellService;
    return undefined;
  },
  interval() {
    throw new Error("Timers are not available to DSH host plugins in P0");
  },
};

const harness = {
  handle(name, handler) {
    const normalized = String(name || "");
    if (!normalized || typeof handler !== "function") throw new Error("Invalid harness.handle registration");
    handlers.set(normalized, handler);
    return () => handlers.delete(normalized);
  },
  defineTool(spec) { return spec; },
  registerTool,
};

globalThis.harness = harness;

function unwrapPlugin(value) {
  let candidate = value;
  if (candidate && typeof candidate === "object" && "default" in candidate) candidate = candidate.default;
  if (candidate && typeof candidate === "object" && "plugin" in candidate) candidate = candidate.plugin;
  return candidate;
}

async function loadPlugin() {
  const entry = path.resolve(sourceRoot, entryPoint);
  if (!entry.startsWith(sourceRoot + path.sep) && entry !== sourceRoot) {
    throw new Error("DSH entry point escapes source root");
  }
  let candidate = await import(pathToFileURL(entry).href);
  candidate = unwrapPlugin(candidate);
  if (typeof candidate === "function") {
    const result = candidate(context, {});
    if (result && typeof result.then === "function") await result;
    return;
  }
  if (!candidate || typeof candidate.apply !== "function") {
    throw new Error("DSH entry must export a function or an object with apply(ctx)");
  }
  const result = candidate.apply(context, {});
  if (result && typeof result.then === "function") await result;
}

function discoveryResult() {
  return {
    protocol_version: VERSION,
    node_version: process.versions.node,
    tools: [...tools.values()].map(({ name, description, parameters_schema }) => ({
      name,
      description,
      parameters_schema,
    })),
    handler_channels: [...handlers.keys()].sort(),
    capabilities: ["compat.shell.run"],
  };
}

async function handleRequest(message) {
  const validObject = message && typeof message === "object" && !Array.isArray(message);
  const requestId = validObject ? String(message.request_id || "") : "";
  try {
    if (!validObject || message.version !== VERSION || message.type !== "request" || !requestId) {
      throw new Error("Invalid DSH bridge request envelope");
    }
    if (!message.payload || typeof message.payload !== "object" || Array.isArray(message.payload)) {
      throw new Error("DSH bridge request payload must be an object");
    }
    if (message.method === "handshake" || message.method === "discover") {
      return { version: VERSION, type: "response", request_id: requestId, ok: true, result: discoveryResult() };
    }
    if (message.method === "invoke") {
      const payload = message.payload || {};
      const tool = tools.get(String(payload.tool_name || ""));
      if (!tool) throw new Error(`Tool not found: ${String(payload.tool_name || "")}`);
      const args = payload.arguments;
      if (!args || typeof args !== "object" || Array.isArray(args)) {
        throw new Error("DSH invoke arguments must be an object");
      }
      const result = await invocationContext.run(requestId, () => tool.execute(args, { requestId }));
      return { version: VERSION, type: "response", request_id: requestId, ok: true, result };
    }
    if (message.method === "shutdown") {
      setImmediate(() => process.exit(0));
      return { version: VERSION, type: "response", request_id: requestId, ok: true, result: "bye" };
    }
    throw new Error(`Unknown DSH bridge method: ${String(message.method || "")}`);
  } catch (error) {
    return {
      version: VERSION,
      type: "response",
      request_id: requestId || "unknown",
      ok: false,
      error: String(error && error.message || error),
      error_type: String(error && error.name || "Error"),
    };
  }
}

function handleCapabilityResponse(message) {
  const pending = pendingCapabilities.get(String(message.request_id || ""));
  if (!pending) return;
  pendingCapabilities.delete(String(message.request_id));
  clearTimeout(pending.timer);
  if (message.ok === true) pending.resolve(message.result);
  else pending.reject(new Error(String(message.error || "Capability failed")));
}

async function boot() {
  await loadPlugin();
  const input = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
  input.on("line", (line) => {
    if (Buffer.byteLength(line, "utf8") > MAX_LINE_BYTES) {
      stderrLine("error", [`Input line exceeds ${MAX_LINE_BYTES} bytes`]);
      process.exitCode = 64;
      input.close();
      return;
    }
    let message;
    try { message = JSON.parse(line); }
    catch (error) {
      stderrLine("error", ["Invalid protocol JSON", error]);
      process.exitCode = 64;
      input.close();
      return;
    }
    if (message.type === "capability_response") {
      handleCapabilityResponse(message);
      return;
    }
    Promise.resolve(handleRequest(message)).then(writeMessage).catch((error) => {
      writeMessage({
        version: VERSION,
        type: "response",
        request_id: String(message && message.request_id || "unknown"),
        ok: false,
        error: String(error && error.message || error),
        error_type: String(error && error.name || "Error"),
      });
    });
  });
}

boot().catch((error) => {
  stderrLine("error", [error && error.stack || error]);
  process.exitCode = 70;
});
