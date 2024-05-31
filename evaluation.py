import easyocr
import os
import logging
from Levenshtein import distance
import nltk
import pandas as pd


# Set up logging
logging.basicConfig(filename='text_detection.log', level=logging.INFO)

# Function to calculate Character Error Rate (CER)
def calculate_cer(gt_text, pred_text):
    gt_text = gt_text.replace(' ', '').lower()
    pred_text = pred_text.replace(' ', '').lower()
    return distance(gt_text, pred_text) / max(len(gt_text), len(pred_text), 1)

# Function to calculate Word Error Rate (WER)
def calculate_wer(gt_text, pred_text):
    gt_words = gt_text.split()
    pred_words = pred_text.split()
    distance = nltk.edit_distance(gt_words, pred_words)
    return distance / max(len(gt_words), len(pred_words), 1)

# Languages and models to use
languages_and_models = ['en']

# Initialize OCR readers for each language/model
readers = [easyocr.Reader([lang]) for lang in languages_and_models]

# Path to your folder containing images
image_folder = r"C:\Users\sreek\Desktop\Mini project 2\OCR images"

ground_truth_mapping = {
    '8ccac23f-dca2-4477-8a74-73283933da6d': "=PTRPWRTRN+TOB2-FC5: 1",
    '39bb0721-4ebf-4e61-a208-53bb97e5f4d2': "=PTRPWRTRN+TOB2-FC5: 1",
    '362a6527-7d14-4c04-92a3-eee8b5c816fb': "=APSELESUP+TOB2-TB1: +",
    '319291cd-6e17-4e4a-95f2-e16c5847b839': "+T0B2-XD0B_3: -: a",
    'b35015ff-8849-4f18-835a-c0b076a69e7b': "=APSELESUP+TOB2-TB1: +",
    'bcbc9729-905e-4735-a8d5-6b241abf92e3': "+TOB2-XDN1:N7: a",
    'c2a6afee-1079-40eb-887d-10f0194d2df6': "+TOB2-XDN1:N7: a",
    'c31a6148-24f7-4759-9211-251f223702bb': "=APSELESUP+T0B2-KF5: 11",
    'cee69595-83fa-48d1-b2e5-ca95c550c6a6': "+T0B2-XDN1:N7: a",
    'd3ee3212-f5c9-47f4-8b66-fcc3ef680364': "=YACYAWACT+TOB3-TA4:PE",
    'd57ac422-bd32-4d21-91aa-37ff60ddfe3b': "=APSELESUP+TOB2-KF5: 11",
    'f16fb7ec-3dc9-4a2b-bb5a-064cd0990df2': "+T0B3-XE1:PE4:b "
}

# Iterate through each image in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(('.png', '.jfif', '.jpg')):
        image_path = os.path.join(image_folder, filename)

        try:
            # Read text from the image using each reader
            combined_results = []
            for reader in readers:
                results = reader.readtext(image_path)
                combined_results.extend(results)

            # Extract ground truth text (if available, replace with actual ground truth)

            ground_truth_text = ground_truth_mapping.get(filename, 'Default ground truth if filename not found')


        #    # Print the extracted text
        #    print(f'Image: {filename}')
        #   for detection in combined_results:
        #       print(detection[1])

            # Calculate CER and WER
        #    cer_values = [calculate_cer(ground_truth_text, detection[1]) for detection in combined_results]
        #    wer_values = [calculate_wer(ground_truth_text, detection[1]) for detection in combined_results]

            # Print CER and WER for each detection
        #    for i, detection in enumerate(combined_results):
        #        print(f"Detection {i + 1}: CER={cer_values[i]}, WER={wer_values[i]}")

            # Log the results
        #    logging.info(f'Image: {filename}, Text: {[detection[1] for detection in combined_results]}, CER: {cer_values}, WER: {wer_values}')

        #except Exception as e:
            # Handle errors
        #    print(f'Error processing image {filename}: {str(e)}')
        #    logging.error(f'Error processing image {filename}: {str(e)}')

        #finally:
        #    print('\n')  # Add a new line between images
            
            cer_value = calculate_cer(ground_truth_text, combined_results[0][1])  # Assuming you are checking against the first detection
            wer_value = calculate_wer(ground_truth_text, combined_results[0][1])  # Similarly, for WER

            # Print or log the CER and WER values
            print(f"Image: {filename}, CER: {cer_value}, WER: {wer_value}")
            logging.info(f"Image: {filename}, CER: {cer_value}, WER: {wer_value}")

        except Exception as e:
            # Handle errors
            print(f'Error processing image {filename}: {str(e)}')
            logging.error(f'Error processing image {filename}: {str(e)}')

        finally:
            print('\n') 