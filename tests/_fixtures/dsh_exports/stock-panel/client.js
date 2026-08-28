return {
  inject: ['timer'],
  apply(ctx) {
    const slots = ctx.get('slots')
    const layout = ctx.get('layout')
    styles.insert('.fixture-stock{display:block}')
    slots.inject('details', function () {
      return slots.register({ name: 'details' }, function () {
        if (layout) layout.openDetails()
        return React.createElement('div', { className: 'fixture-stock' }, 'stock')
      })
    })
  }
}
