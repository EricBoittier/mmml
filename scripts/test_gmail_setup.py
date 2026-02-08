#!/usr/bin/env python3
"""
Test script to verify Gmail API setup.

This script tests if the Gmail API authentication works correctly.
Run this after setting up credentials.json.
"""

from gmail_attachment_crawler import GmailAttachmentCrawler


def test_authentication():
    """Test Gmail API authentication."""
    print("Testing Gmail API authentication...")

    crawler = GmailAttachmentCrawler()

    if crawler.authenticate():
        print("✅ Authentication successful!")

        # Test search (will return empty if no messages found)
        try:
            messages = crawler.search_emails('affidavit')
            print(f"✅ Search successful! Found {len(messages)} messages with 'affidavit'")
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return False

        return True
    else:
        print("❌ Authentication failed!")
        print("Make sure credentials.json is in the project root.")
        return False


if __name__ == '__main__':
    test_authentication()
