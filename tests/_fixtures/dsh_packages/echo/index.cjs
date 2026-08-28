module.exports = {
  apply(ctx) {
    harness.registerTool(ctx, harness.defineTool({
      name: 'fixture_echo',
      description: 'Return the supplied text from a pre-built DSH package.',
      parameters: {
        text: { type: 'string', required: true, description: 'Text to echo' }
      },
      async execute(args) {
        return { ok: true, echo: String(args.text || '') }
      }
    }))
  }
}
