export function renderTextHook(panel, plan, save, disabled = false) {
  panel.replaceChildren();
  panel.hidden = !plan;
  if (!plan) return;
  const node = (tag, text) => {
    const element = document.createElement(tag);
    if (text) element.textContent = text;
    return element;
  };
  const heading = node('h2', 'Text hook');
  heading.id = 'text-hook-heading';
  panel.append(heading);
  if (plan.chosen) {
    panel.append(node('p', plan.chosen));
    return;
  }
  panel.append(node('p', 'Pick an idea or write your own.'));
  const choices = node('div');
  choices.className = 'text-hook-ideas';
  const label = node('label', 'Your text');
  label.setAttribute('for', 'text-hook-text');
  const input = node('textarea');
  input.id = 'text-hook-text';
  input.rows = 2;
  input.maxLength = 80;
  const use = node('button', 'Use this');
  use.type = 'button';
  use.disabled = true;
  const error = node('p');
  error.setAttribute('role', 'alert');
  error.className = 'form-error';
  let source = 'typed';
  const buttons = plan.ideas.map((idea) => {
    const button = node('button', idea);
    button.type = 'button';
    button.disabled = disabled;
    button.setAttribute('aria-pressed', 'false');
    button.addEventListener('click', () => {
      input.value = idea;
      source = 'idea';
      for (const item of buttons) item.setAttribute('aria-pressed', String(item === button));
      use.disabled = disabled;
    });
    return button;
  });
  input.disabled = disabled;
  input.addEventListener('input', () => {
    source = 'typed';
    for (const item of buttons) item.setAttribute('aria-pressed', 'false');
    use.disabled = disabled || !input.value.trim() || input.value.trim().length > 80;
  });
  use.addEventListener('click', async () => {
    const text = input.value.trim();
    if (!text || disabled || use.disabled) return;
    use.disabled = true;
    input.disabled = true;
    for (const button of buttons) button.disabled = true;
    try {
      await save({text_hook: {text, source}});
      renderTextHook(panel, {...plan, chosen: text, source}, save, disabled);
    } catch (failure) {
      error.textContent = failure.message;
      use.disabled = false;
      input.disabled = false;
      for (const button of buttons) button.disabled = false;
    }
  });
  choices.append(...buttons);
  panel.append(choices, label, input, use, error);
}
