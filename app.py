"""Gradio demo:  python app.py   (then open the printed local URL; works offline)."""
import gradio as gr
from predict import assess

def run(img):
    if img is None: return "Upload an image.", ""
    r = assess(img)
    prob = "-" if r["prob"] is None else f"{r['prob']:.3f}"
    banner = {"normal": "NORMAL", "low_confidence": "LOW CONFIDENCE - refer to a clinician", "pneumonia": "PNEUMONIA suspected - refer to a clinician",
              "reject": "CANNOT ASSESS - " + r["note"]}[r["zone"]]
    return banner, f"probability of pneumonia: {prob}\n{r['note']}\nScreening aid only; not a diagnosis."

demo = gr.Interface(run, gr.Image(type="pil", label="Chest X-ray"), [gr.Textbox(label="Result"), gr.Textbox(label="Details")],
                    title="MediScan (Arogya Diagnostics) - paediatric chest X-ray screening aid", allow_flagging="never")
if __name__ == "__main__":
    demo.launch()
