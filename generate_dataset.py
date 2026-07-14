"""Generate a realistic phishing email dataset for training.

Creates dataset/phishing_emails.csv with two columns:
    email_text  - the raw email body
    label       - 1 for phishing, 0 for safe (legitimate)

Run once before training:  python generate_dataset.py
"""

import csv
import os
import random

random.seed(42)

DATASET_PATH = os.path.join("dataset", "phishing_emails.csv")

# Phishing templates (label = 1)
PHISHING_TEMPLATES = [
    "Dear customer, your account at {bank} has been suspended. Verify your identity immediately at {url} or your account will be closed permanently.",
    "You have won a lottery of $1,000,000! To claim your prize, click here {url} and enter your bank details within 24 hours.",
    "Urgent: We detected unusual activity on your {bank} account. Please confirm your password at {url} to restore access.",
    "Your {service} password will expire today. Reset it now at {url} to avoid losing your account.",
    "Dear user, your package could not be delivered. Click {url} to update your shipping address and pay a small fee.",
    "You have been selected for a $500 Amazon gift card. Claim it now at {url} before it expires. Limited time offer!",
    "Security alert: Someone is trying to access your {service} account from a new device. Verify it is you at {url}.",
    "Congratulations! You are approved for a loan of $50,000 with no credit check. Apply now at {url}.",
    "Dear {bank} customer, we need to verify your billing information. Please log in at {url} and confirm your details.",
    "Your PayPal account has been limited. Restore access by confirming your identity at {url}. Act now!",
    "You have an unpaid invoice of $299. Click {url} to view and pay the invoice to avoid legal action.",
    "Dear employee, your salary has been processed. Check your payroll statement at {url} and verify your bank details.",
    "Your Netflix subscription has expired. Renew now at {url} to continue watching. Your account will be suspended.",
    "IRS Notice: You have a tax refund of $2,400 waiting. Click {url} to claim your refund before the deadline.",
    "Dear customer, your {service} subscription will be auto-renewed for $199. To cancel, visit {url}.",
    "You have received a secure document. Open it at {url} and log in with your email credentials to view.",
    "Alert: Your computer has been infected with 3 viruses. Click {url} to download the security tool and remove them.",
    "Dear user, you have been gifted 1 Bitcoin. Claim it at {url} by entering your wallet passphrase.",
    "Your {bank} online banking is locked. Unlock it at {url} with your social security number and date of birth.",
    "You have a new message from {service}. Read it at {url} and log in with your email and password.",
]

# Safe / legitimate templates (label = 0)
SAFE_TEMPLATES = [
    "Hi team, just a reminder that the project meeting is scheduled for tomorrow at 10 AM in conference room B. Please bring your reports.",
    "Dear {name}, thank you for your purchase from {store}. Your order #4521 has been shipped and will arrive in 3-5 business days.",
    "Hello everyone, please find attached the minutes from last week's team meeting. Let me know if you have any questions.",
    "Hi {name}, I hope you are doing well. Can we reschedule our lunch to next Thursday? Let me know what works for you.",
    "Dear customer, your {service} subscription has been successfully renewed. Your next billing date is the 15th of next month.",
    "Good morning, the weekly status report has been uploaded to the shared drive. Please review it before the call.",
    "Hi {name}, thanks for reaching out. I have attached the documents you requested. Feel free to contact me if you need anything else.",
    "Dear all, the office will be closed on Monday for the holiday. We will resume normal operations on Tuesday.",
    "Hello {name}, your appointment with Dr. Smith is confirmed for March 15 at 2 PM. Please arrive 10 minutes early.",
    "Team, please remember to submit your timesheets by Friday end of day. Reach out to HR if you have any issues.",
    "Hi {name}, it was great catching up with you yesterday. Let's plan another coffee soon. Best regards.",
    "Dear {name}, your library book is due in 3 days. You can renew it online or visit the library desk.",
    "Hello, the quarterly newsletter has been published. You can read it on the company intr portal under News.",
    "Hi everyone, the potluck lunch is this Friday. Please sign up on the sheet in the break room. Thanks!",
    "Dear {name}, your application has been received and is under review. We will contact you within 5 business days.",
    "Good afternoon, the new software update is available. Please install it at your earliest convenience. No action needed from your side.",
    "Hi {name}, thank you for attending the webinar. The slides and recording have been emailed to you.",
    "Dear customer, this is a courtesy reminder that your annual warranty expires next month. Contact support for renewal options.",
    "Hello team, great work on the release this week. Let's keep the momentum going. Have a good weekend!",
    "Hi {name}, the photos from the trip are uploaded to the shared album. Feel free to add yours as well.",
]

# Suspicious URLs and domains commonly seen in phishing
SUSPICIOUS_URLS = [
    "http://secure-login-verify.com/login",
    "http://paypal-verify-account.tk",
    "http://bank0famerica-secure.ml",
    "http://amaz0n-prize-claim.cf",
    "http://bit.ly/verify-now",
    "http://tinyurl.com/account-confirm",
    "http://login-secure-update.ga",
    "http://appleid-locked-account.gq",
    "http://netflx-billing-update.tk",
    "http://irs-refund-claim.ml",
    "http://secure-paypal-login.cf",
    "http://amaz0n-secure-login.gq",
    "http://verify-your-account-now.tk",
    "http://login-update-security.ml",
    "http://claim-your-prize-here.cf",
]

# Legitimate-looking domains for safe emails
SAFE_URLS = [
    "https://www.company-portal.com/news",
    "https://www.library-portal.com/renew",
    "https://www.company-portal.com/timesheets",
    "https://www.support-center.com/warranty",
    "https://www.webinar-portal.com/slides",
    "https://www.company-portal.com/shared-drive",
]

BANKS = ["Chase", "Wells Fargo", "Citibank", "HSBC", "Barclays", "Bank of America"]
SERVICES = ["Netflix", "Amazon", "Apple", "Google", "Microsoft", "Facebook", "Instagram"]
STORES = ["Amazon", "eBay", "Best Buy", "Target", "Walmart"]
NAMES = ["John", "Sarah", "Michael", "Emily", "David", "Jessica", "Robert", "Lisa"]


def build_phishing_email():
    template = random.choice(PHISHING_TEMPLATES)
    url = random.choice(SUSPICIOUS_URLS)
    return template.format(
        bank=random.choice(BANKS),
        service=random.choice(SERVICES),
        url=url,
    )


def build_safe_email():
    template = random.choice(SAFE_TEMPLATES)
    if "{url}" in template:
        url = random.choice(SAFE_URLS)
        return template.format(
            name=random.choice(NAMES),
            store=random.choice(STORES),
            service=random.choice(SERVICES),
            url=url,
        )
    return template.format(
        name=random.choice(NAMES),
        store=random.choice(STORES),
        service=random.choice(SERVICES),
    )


def main():
    os.makedirs("dataset", exist_ok=True)
    rows = []

    # Generate 120 phishing emails
    for _ in range(120):
        rows.append((build_phishing_email(), 1))

    # Generate 120 safe emails
    for _ in range(120):
        rows.append((build_safe_email(), 0))

    random.shuffle(rows)

    with open(DATASET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["email_text", "label"])
        writer.writerows(rows)

    phishing_count = sum(1 for _, label in rows if label == 1)
    safe_count = len(rows) - phishing_count
    print(f"Dataset generated: {DATASET_PATH}")
    print(f"Total emails: {len(rows)}  (Phishing: {phishing_count}, Safe: {safe_count})")


if __name__ == "__main__":
    main()
