Intelligent Multi-Class Customer Support Ticket Classifier

Project scaffold for a production-ready NLP system that classifies customer support tickets into categories such as Billing, Technical Issue, Account Access, Product Inquiry, Complaint, and Feedback.

See the `src/`, `api/`, and `dashboard/` folders for core code. Use `docker-compose up --build` to run the API and dashboard locally.

Architecture:

```mermaid
graph LR
  A[CSV Dataset] --> B[Preprocessing]
  B --> C[Baseline Models]
  B --> D[Transformer Fine-tuning]
  C --> E[Evaluation]
  D --> E
  E --> F[Model Registry (MLflow)]
  F --> G[API (FastAPI)]
  F --> H[Dashboard (Streamlit)]
  G --> I[Clients]
  H --> I
```

For setup, see `requirements.txt` and `Dockerfile`.
