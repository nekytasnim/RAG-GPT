import gradio as gr

class UISettings:
  @staticmethod
  def toggle_sidebar(state: bool):
    state = not state
    return gr.update(visible=state), state

  @staticmethod
  def feedback(data: gr.LikeData) -> None:
    value = data.value
    if isinstance(value, dict):
      content = value.get("value", str(value))
    else:
      content = str(value)

    action = "Upvoted" if data.liked else "Downvoted"
    print(f"You {action} this response: {content}")