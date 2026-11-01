from fastapi import FastAPI , Request 
from pydantic import BaseModel   # the request must in a perticular form like : text
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates   # UI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


# initializr Fast API app
app = FastAPI(title = "Text Summarizer App" , description="Text Summarizer using T5" , version="1.0")

# load Model & Tokenizer 
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

# device 
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

# Templating (means where the html code exist)
templates = Jinja2Templates(directory=".")   # . means its in the same file

# input schema for dialogue -> string
class DialogueInput(BaseModel):
    dialogue : str              # just defining that dialogue will be a string value

# cleaning data
def clean_data(text):
  text = re.sub(r"\r\n" , " " , text)   # lines (next line tag)
  text = re.sub(r"\s+" , " " , text)    # spaces
  text = re.sub(r"<.*?>" , " " , text)  # html tags <p> <h1>
  # strip to delete the extra spaces
  text = text.strip().lower()
  return text


# summariztion Function 
def summarize_dialogue(dialogue : str) -> str:
  dialogue = clean_data(dialogue)   # clean the dailogue by using the clean_data func we have created above

  # tokenize
  inputs = tokenizer(
      dialogue,
      padding = "max_length",
      max_length = 512,
      truncation = True,
      return_tensors = "pt"   # it will return pytorch tensors & if we are using tensorflow so we write tf
  ).to(device)

  # generate the summary --- which will be in token ids or tokens
  model.to(device)
  targets = model.generate(
      input_ids = inputs["input_ids"],
      attention_mask = inputs["attention_mask"],
      max_length = 150,
      num_beams = 4,   # 4 diff summary sequences or generate 4 diff outputs and compare them and then give the answer (to polish output, perfomance good)
      early_stopping = True     # when 4 outputs comes then it will be the EOS(End of sequence) & we will stop
  )

  # so we convert token ids to text (summary) --decoding
  summary = tokenizer.decode(targets[0] , skip_special_tokens=True)  # EOS , seperations , tags will be skip
  return summary

# API End Points:   (GET : /  ,  POST : /summarize/)

@app.post("/summarize/")
async def summarize(dialogue_input : DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary" : summary}

@app.get("/" , response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
    request=request,
    name="index.html",
    context={}
)