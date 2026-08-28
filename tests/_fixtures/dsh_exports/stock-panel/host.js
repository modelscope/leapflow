return {
  apply(ctx) {
    async function fetchQuote(symbol) {
      const shell = ctx.get('shell')
      if (shell === undefined) return { ok: false, error: 'shell unavailable' }
      const command = "curl -sS -m 5 'https://example.test/quote?q=" + symbol + "'"
      const result = await shell.run(shell.resolve({ command, timeoutMs: 5000, stdoutMaxBytes: 4096 }))
      if (result.exitCode !== 0) return { ok: false, error: result.stderr.text }
      return { ok: true, symbol, raw: result.stdout.text }
    }

    harness.handle('fetch-quote', async (args) => fetchQuote(String(args.symbol || 'AAPL')))
    harness.registerTool(ctx, harness.defineTool({
      name: 'fixture_stock_quote',
      description: 'Fetch one fixture stock quote through the typed HTTP capability.',
      parameters: {
        symbol: { type: 'string', required: true, description: 'Ticker symbol' }
      },
      async execute(args) { return fetchQuote(String(args.symbol || 'AAPL')) }
    }))
  }
}
