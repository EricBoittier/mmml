#!/usr/bin/env python3
"""
Gmail Attachment Crawler

This script uses the Google Mail API to search for emails containing 'affidavit',
download their attachments, and create a CSV file with date, event, and source information.
"""

import os
import base64
import csv
import datetime
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


class GmailAttachmentCrawler:
    """Crawler for downloading Gmail attachments and extracting data."""

    SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

    def __init__(self, credentials_path: str = 'credentials.json', token_path: str = 'token.json'):
        """
        Initialize the Gmail crawler.

        Args:
            credentials_path: Path to OAuth 2.0 credentials file
            token_path: Path to store/retrieve access token
        """
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.service = None

    def authenticate(self) -> bool:
        """
        Authenticate with Google Mail API.

        Returns:
            bool: True if authentication successful, False otherwise
        """
        creds = None

        # Load existing token if available
        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, self.SCOPES)

        # If there are no (valid) credentials available, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_path):
                    print(f"Error: {self.credentials_path} not found.")
                    print("Please download credentials.json from Google Cloud Console.")
                    return False

                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, self.SCOPES)
                creds = flow.run_local_server(port=0)

            # Save the credentials for the next run
            with open(self.token_path, 'w') as token:
                token.write(creds.to_json())

        try:
            self.service = build('gmail', 'v1', credentials=creds)
            return True
        except Exception as e:
            print(f"Authentication failed: {e}")
            return False

    def search_emails(self, query: str = 'affidavit') -> List[Dict]:
        """
        Search for emails matching the query.

        Args:
            query: Search query for Gmail

        Returns:
            List of message dictionaries
        """
        if not self.service:
            raise RuntimeError("Not authenticated. Call authenticate() first.")

        try:
            results = self.service.users().messages().list(
                userId='me',
                q=query,
                maxResults=500  # Adjust as needed
            ).execute()

            messages = results.get('messages', [])
            return messages

        except HttpError as error:
            print(f'An error occurred: {error}')
            return []

    def get_message_details(self, message_id: str) -> Optional[Dict]:
        """
        Get full message details including attachments.

        Args:
            message_id: Gmail message ID

        Returns:
            Message details dictionary or None if error
        """
        try:
            message = self.service.users().messages().get(
                userId='me',
                id=message_id,
                format='full'
            ).execute()

            return message

        except HttpError as error:
            print(f'Error getting message {message_id}: {error}')
            return None

    def download_attachment(self, message_id: str, attachment_id: str, filename: str) -> Optional[bytes]:
        """
        Download an attachment from a message.

        Args:
            message_id: Gmail message ID
            attachment_id: Attachment ID
            filename: Filename for the attachment

        Returns:
            Attachment data as bytes or None if error
        """
        try:
            attachment = self.service.users().messages().attachments().get(
                userId='me',
                messageId=message_id,
                id=attachment_id
            ).execute()

            data = attachment.get('data', '')
            return base64.urlsafe_b64decode(data)

        except HttpError as error:
            print(f'Error downloading attachment {filename}: {error}')
            return None

    def extract_data_from_attachments(self, messages: List[Dict], output_dir: str = 'attachments') -> List[Dict]:
        """
        Extract data from email attachments and create records.

        Args:
            messages: List of Gmail message dictionaries
            output_dir: Directory to save attachments

        Returns:
            List of records with date, event, source
        """
        records = []
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        for message in messages:
            message_id = message['id']
            message_details = self.get_message_details(message_id)

            if not message_details:
                continue

            # Extract message metadata
            headers = message_details.get('payload', {}).get('headers', [])
            subject = ''
            date_str = ''

            for header in headers:
                if header['name'].lower() == 'subject':
                    subject = header['value']
                elif header['name'].lower() == 'date':
                    date_str = header['value']

            # Parse date
            try:
                # Gmail date format: "Wed, 30 Dec 2025 10:30:00 +0000"
                date_obj = datetime.datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %z')
                date_formatted = date_obj.strftime('%Y-%m-%d')
            except:
                date_formatted = date_str  # Fallback to original string

            # Process attachments
            parts = message_details.get('payload', {}).get('parts', [])
            for part in parts:
                if part.get('filename'):
                    attachment_id = part['body'].get('attachmentId')
                    filename = part['filename']

                    if attachment_id:
                        attachment_data = self.download_attachment(message_id, attachment_id, filename)

                        if attachment_data:
                            # Save attachment
                            file_path = output_path / f"{message_id}_{filename}"
                            with open(file_path, 'wb') as f:
                                f.write(attachment_data)

                            # Create record (this is a basic implementation)
                            # You'll need to customize this based on your attachment content
                            record = {
                                'date': date_formatted,
                                'event': subject,
                                'source': filename,
                                'attachment_path': str(file_path)
                            }
                            records.append(record)

        return records

    def create_csv(self, records: List[Dict], output_file: str = 'gmail_attachments.csv'):
        """
        Create CSV file from records.

        Args:
            records: List of record dictionaries
            output_file: Output CSV filename
        """
        if not records:
            print("No records to write to CSV.")
            return

        df = pd.DataFrame(records)
        df.to_csv(output_file, index=False)
        print(f"Created CSV file: {output_file} with {len(records)} records")


def main():
    """Main function to run the Gmail attachment crawler."""
    crawler = GmailAttachmentCrawler()

    # Authenticate
    if not crawler.authenticate():
        return

    # Search for emails with 'affidavit'
    print("Searching for emails containing 'affidavit'...")
    messages = crawler.search_emails('affidavit')
    print(f"Found {len(messages)} messages.")

    if not messages:
        print("No messages found.")
        return

    # Extract data from attachments
    print("Downloading attachments and extracting data...")
    records = crawler.extract_data_from_attachments(messages)

    # Create CSV
    if records:
        crawler.create_csv(records)
        print("Done!")
    else:
        print("No attachments found or processed.")


if __name__ == '__main__':
    main()
