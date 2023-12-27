from INSTA.myModule import myModule
import torch

test = myModule()
support_data = torch.randn([1,5,640,5,5]).to("cuda")
query_data = torch.randn([1, 75, 640,5,5]).to("cuda")

support_data,query_data = test.forward(support_data,query_data)
print(support_data.size())
print(query_data.size())