# Gmail Attachment Crawler

This tool uses the Google Mail API to search for emails containing "affidavit", download their attachments, and create a CSV file with date, event, and source information.

## Setup Instructions

### 1. Google Cloud Console Setup

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Gmail API:
   - Go to "APIs & Services" > "Library"
   - Search for "Gmail API" and enable it

### 2. Create OAuth 2.0 Credentials

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth 2.0 Client IDs"
3. Configure the OAuth consent screen if prompted
4. Select "Desktop application" as the application type
5. Download the credentials JSON file
6. Rename it to `credentials.json` and place it in the project root directory

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
cd /path/to/mmml
python scripts/gmail_attachment_crawler.py
```

### Advanced Usage

You can also import and use the crawler programmatically:

```python
from scripts.gmail_attachment_crawler import GmailAttachmentCrawler

crawler = GmailAttachmentCrawler()

# Authenticate
if crawler.authenticate():
    # Search for emails
    messages = crawler.search_emails('affidavit')

    # Extract data and create CSV
    records = crawler.extract_data_from_attachments(messages)
    crawler.create_csv(records, 'output.csv')
```

## Output

The script will create:
- `gmail_attachments.csv`: CSV file with columns: date, event, source, attachment_path
- `attachments/`: Directory containing downloaded attachment files

## CSV Format

The output CSV contains the following columns:
- `date`: Email date (YYYY-MM-DD format)
- `event`: Email subject line
- `source`: Attachment filename
- `attachment_path`: Local path to downloaded attachment

## Customization

The data extraction logic in `extract_data_from_attachments()` is basic and may need customization based on your specific attachment types and data requirements. You can modify this method to:

- Parse specific file formats (PDF, DOCX, etc.)
- Extract structured data from attachments
- Apply custom logic for determining event and source information

## Security Notes

- The script requires read-only access to Gmail
- Credentials are stored locally in `token.json` after first authentication
- Never commit `credentials.json` or `token.json` to version control
- Consider adding these files to `.gitignore`
