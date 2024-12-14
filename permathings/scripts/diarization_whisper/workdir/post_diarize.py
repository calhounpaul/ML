import torch
import os
import json
import time
import logging
from transformers import LlamaForCausalLM, AutoTokenizer
from diarizationlm import utils

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class SecondPassDiarizer:
    def __init__(self, model_name="google/DiarizationLM-8b-Fisher-v2", max_chunk_words=500):
        logger.info("Initializing DiarizationLM...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.device_map = "auto"
        self.max_chunk_words = max_chunk_words
        
        # Load tokenizer
        start_time = time.time()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Tokenizer loaded in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise
        
        # Load model
        start_time = time.time()
        try:
            self.model = LlamaForCausalLM.from_pretrained(
                model_name,
                device_map=self.device_map,
                torch_dtype=torch.float16  # Add this to reduce memory usage
            )
            logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def preserve_speaker_numbers(self, original_text, processed_text):
        """
        Map speaker numbers from processed text back to original speaker numbers.
        """
        # Extract original speaker numbers
        original_speakers = set()
        for part in original_text.split('<speaker:'):
            if part:
                number = part.split('>')[0].strip()
                if number.isdigit():
                    original_speakers.add(int(number))
        original_speakers = sorted(list(original_speakers))

        # Extract processed speaker numbers
        processed_speakers = set()
        for part in processed_text.split('<speaker:'):
            if part:
                number = part.split('>')[0].strip()
                if number.isdigit():
                    processed_speakers.add(int(number))
        processed_speakers = sorted(list(processed_speakers))

        # Create mapping from processed to original
        speaker_map = {}
        for proc, orig in zip(processed_speakers, original_speakers):
            speaker_map[str(proc)] = str(orig).zfill(2)  # Pad with zeros for consistent formatting

        # Replace speaker numbers in processed text
        result = processed_text
        for proc, orig in speaker_map.items():
            result = result.replace(f'<speaker:{proc}>', f'<speaker:{orig}>')

        return result

    def process_chunk(self, chunk_text):
        """Process a single chunk through the DiarizationLM model."""
        logger.info(f"Processing chunk of length {len(chunk_text)} characters...")
        
        try:
            # Add debugging information
            logger.info("Preparing input for tokenizer...")
            input_text = chunk_text + " --> "
            
            # Tokenize with error handling
            try:
                inputs = self.tokenizer([input_text], return_tensors="pt").to(self.device)
                logger.info(f"Input tokenized successfully, shape: {inputs.input_ids.shape}")
            except Exception as e:
                logger.error(f"Tokenization error: {e}")
                raise
            
            # Generate with error handling
            try:
                logger.info("Starting generation...")
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=int(inputs.input_ids.shape[1] * 1.2),
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_beams=1
                )
                logger.info(f"Generation complete, output shape: {outputs.shape}")
            except Exception as e:
                logger.error(f"Generation error: {e}")
                raise
            
            # Decode with error handling
            try:
                logger.info("Decoding output...")
                completion = self.tokenizer.batch_decode(
                    outputs[:, inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                )[0]
                logger.info(f"Decoded completion length: {len(completion)}")
            except Exception as e:
                logger.error(f"Decoding error: {e}")
                raise
            
            # Transfer completion and preserve speaker numbers
            try:
                logger.info("Transferring completion...")
                result = utils.transfer_llm_completion(completion, chunk_text)
                result = self.preserve_speaker_numbers(chunk_text, result)
                logger.info(f"Transfer complete, result length: {len(result)}")
                return result
            except Exception as e:
                logger.error(f"Transfer error: {e}")
                raise
            
        except Exception as e:
            logger.error(f"Error in process_chunk: {e}")
            raise

    def process_file_pair(self, rttm_path, json_path, output_path):
        """Process a pair of RTTM and JSON files to create second-pass diarization."""
        logger.info(f"\nProcessing new file pair:")
        logger.info(f"RTTM: {rttm_path}")
        logger.info(f"JSON: {json_path}")
        
        try:
            # Read input files
            segments = self.read_rttm_file(rttm_path)
            words = self.read_whisper_json(json_path)
            
            # Split into chunks
            chunks = self.chunk_words_and_segments(words, segments)
            logger.info(f"Split into {len(chunks)} chunks")
            
            # Process each chunk
            processed_chunks = []
            for i, chunk in enumerate(chunks, 1):
                logger.info(f"Processing chunk {i}/{len(chunks)}")
                
                try:
                    # Align transcripts with speakers for this chunk
                    chunk_text = self.align_chunk_transcripts(chunk)
                    logger.info(f"Chunk {i} text preview: {chunk_text[:100]}...")
                    
                    # Process through DiarizationLM
                    processed_text = self.process_chunk(chunk_text)
                    
                    # Save chunk result immediately
                    chunk_result = {
                        'start_time': chunk['start_time'],
                        'end_time': chunk['end_time'],
                        'original_text': chunk_text,
                        'processed_text': processed_text
                    }
                    processed_chunks.append(chunk_result)
                    
                    # Write intermediate results
                    temp_output_path = output_path + f'.temp_{i}'
                    with open(temp_output_path, 'w') as f:
                        json.dump({
                            'chunks_processed': i,
                            'total_chunks': len(chunks),
                            'chunks': processed_chunks
                        }, f, indent=2)
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {e}")
                    continue
            
            # Combine processed chunks
            final_output = ' '.join(chunk['processed_text'] for chunk in processed_chunks)
            
            # Save final output
            with open(output_path, 'w') as f:
                json.dump({
                    'final_diarization': final_output,
                    'chunks': processed_chunks
                }, f, indent=2)
            
            # Clean up temp files
            for i in range(1, len(chunks) + 1):
                temp_file = output_path + f'.temp_{i}'
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            logger.info(f"Saved output to: {output_path}")
            return final_output
            
        except Exception as e:
            logger.error(f"Error in process_file_pair: {e}")
            raise

    def read_rttm_file(self, rttm_path):
        """Read RTTM file and return list of speaker segments."""
        segments = []
        with open(rttm_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8 and parts[0] == "SPEAKER":
                    segment = {
                        'start': float(parts[3]),
                        'duration': float(parts[4]),
                        'end': float(parts[3]) + float(parts[4]),
                        'speaker': parts[7].replace('SPEAKER_', '')
                    }
                    segments.append(segment)
        return sorted(segments, key=lambda x: x['start'])

    def read_whisper_json(self, json_path):
        """Read Whisper JSON output and return word-level timestamps."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data.get('chunks', [])

    def merge_adjacent_segments(self, segments, time_threshold=0.5):
        """Merge adjacent segments from the same speaker if they're close in time."""
        if not segments:
            return segments
        
        merged = []
        current = segments[0].copy()
        
        for next_seg in segments[1:]:
            if (next_seg['speaker'] == current['speaker'] and 
                next_seg['start'] - current['end'] <= time_threshold):
                current['end'] = next_seg['end']
                current['duration'] = current['end'] - current['start']
            else:
                merged.append(current)
                current = next_seg.copy()
        
        merged.append(current)
        return merged

    def chunk_words_and_segments(self, words, segments):
        """Split words and segments into chunks while maintaining speaker context."""
        chunks = []
        current_chunk_words = []
        current_chunk_segments = []
        current_word_count = 0
        last_end_time = 0
        
        for word in words:
            start_time, end_time = word.get('timestamp', (0, 0))
            if isinstance(start_time, str) or isinstance(end_time, str):
                continue
                
            # Get relevant segments for this word
            while segments and segments[0]['end'] < start_time:
                segments.pop(0)
            
            current_segments = [s for s in segments if s['start'] <= end_time and s['end'] >= start_time]
            
            current_chunk_words.append(word)
            current_chunk_segments.extend(s for s in current_segments if s not in current_chunk_segments)
            current_word_count += 1
            
            # Create new chunk if we hit the word limit or have no more segments
            if current_word_count >= self.max_chunk_words or not segments:
                if current_chunk_words:
                    chunks.append({
                        'words': current_chunk_words,
                        'segments': sorted(current_chunk_segments, key=lambda x: x['start']),
                        'start_time': current_chunk_words[0]['timestamp'][0],
                        'end_time': current_chunk_words[-1]['timestamp'][1]
                    })
                    # Keep last few words for context in next chunk
                    overlap_words = current_chunk_words[-50:] if len(current_chunk_words) > 50 else current_chunk_words
                    current_chunk_words = overlap_words
                    current_word_count = len(overlap_words)
                    last_end_time = chunks[-1]['end_time']
                    
                    # Keep segments that might be relevant for next chunk
                    current_chunk_segments = [s for s in current_chunk_segments 
                                           if s['end'] > last_end_time]
        
        return chunks

    def align_chunk_transcripts(self, chunk):
        """Align transcripts with speakers for a single chunk."""
        words = chunk['words']
        segments = chunk['segments']
        formatted_text = []
        current_speaker = None
        buffer_words = []
        
        def flush_buffer():
            if buffer_words:
                formatted_text.extend([" ".join(buffer_words), " "])
                buffer_words.clear()
        
        for word in words:
            start_time = word['timestamp'][0]
            
            # Find the corresponding speaker segment
            current_segment = None
            for segment in segments:
                if start_time >= segment['start'] and start_time <= segment['end']:
                    current_segment = segment
                    break
            
            if current_segment:
                speaker = f"speaker:{current_segment['speaker']}"
                if speaker != current_speaker:
                    flush_buffer()
                    if formatted_text:
                        formatted_text.append(" ")
                    formatted_text.extend([f"<{speaker}>", " "])
                    current_speaker = speaker
                buffer_words.append(word['text'].strip())
        
        flush_buffer()
        return "".join(formatted_text).strip()

def main():
    logger.info("Starting second-pass diarization process...")
    
    WORKDIR_PATH = "/workdir"
    RTTM_PATH = os.path.join(WORKDIR_PATH, "diarization_output_files")
    JSON_PATH = os.path.join(WORKDIR_PATH, "whisper_output_files")
    OUTPUT_PATH = os.path.join(WORKDIR_PATH, "second_pass_output")
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    diarizer = SecondPassDiarizer(max_chunk_words=500)  # Process 500 words at a time
    
    total_files = 0
    processed_files = 0
    skipped_files = 0
    error_files = 0
    
    for root, _, files in os.walk(RTTM_PATH):
        for file in files:
            if file.endswith('.rttm'):
                total_files += 1
                
                rttm_file = os.path.join(root, file)
                relative_path = os.path.relpath(root, RTTM_PATH)
                json_file = os.path.join(JSON_PATH, relative_path, 
                                       file.replace('.rttm', '.mp3.json'))
                output_file = os.path.join(OUTPUT_PATH, relative_path,
                                         file.replace('.rttm', '_second_pass.json'))
                
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                
                if os.path.exists(output_file):
                    logger.info(f"Skipping existing file: {output_file}")
                    skipped_files += 1
                    continue
                
                try:
                    if os.path.exists(json_file):
                        diarizer.process_file_pair(rttm_file, json_file, output_file)
                        processed_files += 1
                        logger.info(f"\nProgress: Processed {processed_files}/{total_files} files "
                                  f"(Skipped: {skipped_files}, Errors: {error_files})")
                    else:
                        logger.warning(f"Missing JSON file for: {rttm_file}")
                        error_files += 1
                except Exception as e:
                    logger.error(f"Error processing {rttm_file}: {e}")
                    error_files += 1
    
    logger.info("\nProcessing complete!")
    logger.info(f"Total files found: {total_files}")
    logger.info(f"Successfully processed: {processed_files}")
    logger.info(f"Skipped existing: {skipped_files}")
    logger.info(f"Errors: {error_files}")

if __name__ == "__main__":
    main()