"""
Structured Data Extraction with AWS Bedrock and Instructor
-------------------------------------------------------

This example demonstrates how to use Instructor with AWS Bedrock to extract structured data
from unstructured text. It shows how to:
1. Set up Instructor with AWS Bedrock
2. Define Pydantic models for structured output
3. Extract and validate data using Claude 3

Prerequisites:
-------------
- AWS credentials configured (either via environment variables or AWS CLI)
  export AWS_ACCESS_KEY_ID=your_access_key
  export AWS_SECRET_ACCESS_KEY=your_secret_key
  export AWS_DEFAULT_REGION=us-west-2

- Required packages:
  pip install boto3 instructor pydantic

Note: This example currently works with Instructor's Boto3 integration branch:
      pip install git+https://github.com/k3nnethfrancis/instructor.git@add-boto3-client
      
      It will work with `pip install instructor` once merged into main.
"""

import json
import logging
from typing import List, Optional
import boto3
import instructor
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for structured output
class Patient(BaseModel):
    """Patient information from prescription."""
    name: Optional[str] = None
    date_of_birth: Optional[str] = None

class Medication(BaseModel):
    """Medication details from prescription."""
    name: Optional[str] = None
    strength: Optional[str] = None
    quantity: Optional[int] = None
    sig: Optional[str] = None
    warnings: Optional[List[str]] = Field(default_factory=list)

class PrescriptionExtraction(BaseModel):
    """Main model for prescription extraction."""
    patient: Patient
    medication: Medication
    date_written: Optional[str] = None

class BedrockExtractor:
    """Handles prescription data extraction using AWS Bedrock with Instructor."""

    def __init__(self):
        """Initialize Bedrock client with Instructor."""
        # Create Bedrock client
        bedrock_client = boto3.client('bedrock-runtime')
        
        # Initialize model ID for Claude 3
        self.model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
        
        # Create the instructor client with BOTO3_TOOLS mode
        self.client = instructor.from_boto3(
            bedrock_client, 
            mode=instructor.Mode.BOTO3_TOOLS
        )

    def extract_prescription_data(self, text: str, entities: list) -> PrescriptionExtraction:
        """
        Extract prescription details using Bedrock and Instructor.

        Args:
            text (str): Raw text from prescription
            entities (list): List of detected medical entities

        Returns:
            PrescriptionExtraction: Structured prescription data
        """
        try:
            prompt = f"""Extract prescription details from the following text and medical entities and return them in a JSON object that exactly matches this schema:

{{
    "patient": {{
        "name": string | null,
        "date_of_birth": string | null
    }},
    "medication": {{
        "name": string | null,
        "strength": string | null,
        "quantity": number | null,
        "sig": string | null,
        "warnings": string[] | null
    }},
    "date_written": string | null
}}

Raw Text from Prescription:
{text}

Medical Entities Detected:
{json.dumps(entities, indent=2)}

Important guidelines:
1. Use null for missing values
2. Return ONLY the JSON object, no additional text
"""
            # Call Bedrock with Claude 3's message format
            response = self.client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 4096,
                    "temperature": 0.1,
                    "top_p": 0.999,
                    "top_k": 250,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ]
                })
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            content = response_body['content'][0]['text']
            
            # Parse JSON response (handle both raw and code-block formats)
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
                if json_match:
                    content = json.loads(json_match.group(1))
                else:
                    raise ValueError("Could not parse JSON from response")
            
            # Validate and return structured data
            return PrescriptionExtraction.model_validate(content)

        except Exception as e:
            logger.error("Error in prescription extraction", exc_info=True)
            raise e

def main():
    """Example usage of the BedrockExtractor."""
    # Sample test data
    sample_data = {
        "raw_text": """
        PATIENT: John Smith
        DOB: 1980-01-01
        
        MEDICATION: Amoxicillin 500mg capsules
        QUANTITY: 30
        SIG: Take 1 capsule by mouth twice daily
        
        WARNINGS: Take with food
        DATE WRITTEN: 2024-03-14
        """,
        "comprehend_entities": [
            {"Text": "Amoxicillin", "Type": "MEDICATION"},
            {"Text": "500mg", "Type": "STRENGTH"},
            {"Text": "John Smith", "Type": "PATIENT"}
        ]
    }

    # Initialize extractor
    extractor = BedrockExtractor()
    
    try:
        # Extract prescription details
        logger.info("Extracting prescription details...")
        result = extractor.extract_prescription_data(
            text=sample_data['raw_text'],
            entities=sample_data['comprehend_entities']
        )
        
        # Print results
        logger.info("\nExtracted Prescription Details:")
        print(json.dumps(result.model_dump(), indent=2))
        
    except Exception as e:
        logger.error(f"Error processing prescription: {e}")

if __name__ == "__main__":
    main()