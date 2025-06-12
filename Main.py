import json
from langgraph.graph import StateGraph
from pydantic import BaseModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
count=0
# --- Training data as a dictionary ---
intent_dict = {
    'ReKYC': [
        "Please send us a valid ID and proof of address.",
        "Your data helps keep your account safe.",
        "update your KYC details"
    ],
    'PolicyRenewal': [
        "Your policy is due for renewal",
        "A quick renewal today keeps you protected",
        "There is a change in our terms and agreements"
    ],
    'ClaimsIntake': [
        "We’ve received your claim request",
        "reviewing your claim now.",
        "Please keep your reference number"
    ]
}

# --- Flatten dictionary into list of emails and labels ---
train_emails = []
train_labels = []
for label, emails in intent_dict.items():
    for email in emails:
        train_emails.append(email)
        train_labels.append(label)

# --- Model setup ---
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_emails)
clf = LogisticRegression()
clf.fit(X_train, train_labels)

class Email_Inten(BaseModel):
    email: str
    intent: str

email_var = StateGraph(Email_Inten)

def classify_node(state: Email_Inten) -> Email_Inten:
    email_val = state.email
    X = vectorizer.transform([email_val])
    pred = clf.predict(X)[0]
    state.intent = pred
    # Commented out to avoid duplicate output
    # print(f"Intent detected: {state.intent}\n")
    return state

def TriggerWorkflow(state: Email_Inten) -> Email_Inten:
    global count
    count+=1
    print(f"⚡ Triggering {state.intent} workflow\n")
    output = {"intent": state.intent, "status": "triggered"}
    print(json.dumps(output))
    
    # Open a file in append mode ('a'), so new entries are added to the end
    with open("emails.csv", "a") as f:
        f.write(f"id: {count}, Email: {state.email}, Intent: {state.intent}\n")
    
    return state

email_var.add_node("classify", classify_node)
email_var.add_node("trigger", TriggerWorkflow)
email_var.add_edge("__start__", "classify")
email_var.add_edge("classify", "trigger")
workflow = email_var.compile()

if __name__ == "__main__":
    while True:
        email_input = input("Email (type 'done' to finish): ")
        if email_input.strip().lower() == "done":
            print("done")
            break
        initial_state = Email_Inten(email=email_input, intent="none")
        workflow.invoke(initial_state)
