import os
from writerai import Writer
writer_api_key = "9JUHgp48WQpJMpYJAqOB0xDPlKzb4EFL"

client = Writer(
    # This is the default and can be omitted
    api_key=writer_api_key,
)
model_list_response = client.models.list()
print(model_list_response.models)