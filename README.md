# Legal Docs Project

This repo focus on using mlops too to maintain, deploy and combine retrieval system with rag and AGENT
our pipeline is

image

# DATA LAKE AND DATA VERSION CONTROL

Before Fine-tunning model, we have to preprocess data extremely carefull because cross-encoder in reranker stage is very sensitive with text data. Therefore, we utilize DvC which is a tool to controll data version and combine it with AWS Bucket S3 to store data version

![DvC CLI](./images/dvc.png)
![AWS S3](./images/AWS3.png.png)
