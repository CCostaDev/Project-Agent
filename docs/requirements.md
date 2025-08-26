**Project Overview**

Project Name: Training Support Chatbot
Purpose: Provide employees with a 24/7 chatbot that answers training and system-related queries using documents stored in SharePoint
Scope: Internal use only, hosted on company Azure environment

**Functional Requirements**

FR1 Document ingestion
[ ] Fetch training documents from a designated SharePoint folder.
[ ] Automatically sync new/updated/deleted files weekly.

FR2 Vectorisation
[ ] Chunk documents, generate embeddings (OpenAI API), store in ChromaDB.

FR3 Chatbot functionality
[ ] Accept natural language questions via Streamlit UI.
[ ] Retrieve relevant document chunks and generate answers with citations.

FR4 Citations
[ ] Display document names and links (webUrl from SharePoint metadata).

FR5 Feedback (optional)
[ ] Allow users to rate answers (👍/👎) for continuous improvement.

**Non-Functional Requirements**

NFR1 Performance
[ ] Average query response < 8 seconds.
[ ] Retrieval step < 3 seconds.

NFR2 Security
[ ] Secrets (API keys, client credentials) stored in Azure Key Vault.
[ ] MSAL OAuth2.0 for Graph API access.
[ ] TLS enforced for all traffic.

NFR3 Reliability
[ ] Weekly sync must complete successfully ≥ 99% of runs.
[ ] Fault tolerance: retries on Graph/OpenAI failures.

NFR4 Usability
[ ] Web UI accessible via browser, simple and intuitive.
[ ] Mobile browser compatible.

NFR5 Maintainability
[ ] Repo structured with clear modules.
[ ] Automated CI/CD pipeline with tests.

NFR6 Accessibility
[ ] Clear text, readable layout (Streamlit).
[ ] Works on standard desktops and laptops used internally.

**Constraints**

- Must use company Azure infrastructure.
- Must only index documents from the approved SharePoint folder.
- Development timeframe: 4 weeks total.

**Dependencies**

- Python
- Streamlit
- LangChain
- OpenAI API
- ChromaDB
- Microsoft Graph API (via MSAL auth)
- Azure Key Vault / App Service / Container Apps

**Success Criteria**

- ≥ 80% of golden Q&A set answered correctly with relevant citations.
- Chatbot deployed internally and accessible to employees.
- Positive user feedback (System Usability Scale ≥ 70).
- Evidence of following SDLC in project report.
