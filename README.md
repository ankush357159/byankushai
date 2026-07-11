### Run project
`uv run uvicorn app.main:app --reload`


### If GPU, run below command to install optimum with onnxruntime-gpu
`uv add "optimum[onnxruntime-gpu]"`

### For CPU, run below command to install optimum with onnxruntime
`uv add "optimum[onnxruntime]"`

