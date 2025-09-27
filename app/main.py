#!/usr/bin/env python3
"""
The WebCast Pipeline Integration
Main entry point that connects all pipeline components according to the README steps.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import all pipeline components
from search.search_trigger import needs_live_data_openai  # pyright: ignore[reportMissingImports]
from search.merged_search import merged_search
from search.webcast_engine_openai import get_llm_summary
from speach.text_to_speech import stream_text_to_speech


def webcast_pipeline(user_prompt: str, enable_voice: bool = True) -> dict:
    """
    Main pipeline function that integrates all components according to README steps.
    
    Args:
        user_prompt: The user's question/prompt
        enable_voice: Whether to generate voice output (default: True)
    
    Returns:
        dict: Contains the text response and optionally the audio file path
    """
    print(f"[WebCast] Processing prompt: {user_prompt}")
    
    # Step 1: Check if we need live data using search trigger
    print("[WebCast] Step 1: Checking if live data is needed...")
    needs_search = needs_live_data_openai(user_prompt)
    print(f"[WebCast] Search needed: {needs_search}")
    
    context = None
    
    # Step 2: If search is needed, get context from web
    if needs_search:
        print("[WebCast] Step 2: Fetching live data from web...")
        try:
            paragraphs = merged_search(user_prompt)
            if paragraphs:
                # Combine the best paragraphs for richer context
                context = " ".join(paragraphs[:2])  # Use top 2 paragraphs for better context
                print(f"[WebCast] Retrieved context: {context[:100]}...")
            else:
                print("[WebCast] No relevant paragraphs found from search")
        except Exception as e:
            print(f"[WebCast] Error during search: {e}")
            # If it's a signal threading error, provide a more helpful message
            if "signal only works in main thread" in str(e) or "signal" in str(e).lower():
                print("[WebCast] Threading issue detected - continuing without live data")
            context = None
    else:
        print("[WebCast] Step 2: Skipping search - using existing knowledge")
    
    # Step 3: Get LLM summary with or without context
    print("[WebCast] Step 3: Generating LLM response...")
    try:
        text_response = get_llm_summary(user_query=user_prompt, context=context)
        print(f"[WebCast] Generated response: {text_response[:100]}...")
    except Exception as e:
        print(f"[WebCast] Error generating LLM response: {e}")
        return {"error": f"Failed to generate response: {e}"}
    
    result = {
        "text_response": text_response,
        "context_used": context is not None,
        "search_performed": needs_search
    }
    
    # Step 4: Generate voice output if requested
    if enable_voice and text_response:
        print("[WebCast] Step 4: Generating voice output...")
        try:
            audio_file = stream_text_to_speech(
                text=text_response,
                save_to_file=True
            )
            if audio_file:
                result["audio_file"] = audio_file
                print(f"[WebCast] Audio saved to: {audio_file}")
            else:
                print("[WebCast] Voice generation completed but no file saved")
        except Exception as e:
            print(f"[WebCast] Error generating voice: {e}")
            result["voice_error"] = str(e)
    
    return result


def main():
    """
    Interactive main function for testing the pipeline.
    """
    print("=== The WebCast Pipeline ===")
    print("Enter your question (or 'quit' to exit):")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                print("Please enter a question.")
                continue
            
            # Ask if user wants voice output
            voice_choice = input("Generate voice output? (y/n, default: y): ").strip().lower()
            enable_voice = voice_choice != 'n'
            
            # Run the pipeline
            result = webcast_pipeline(user_input, enable_voice=enable_voice)
            
            # Display results
            print("\n" + "="*50)
            print("RESPONSE:")
            print(result["text_response"])
            print("="*50)
            
            if result.get("context_used"):
                print("✓ Used live web data")
            else:
                print("✓ Used existing knowledge")
            
            if result.get("search_performed"):
                print("✓ Performed web search")
            else:
                print("✓ No search needed")
            
            if result.get("audio_file"):
                print(f"✓ Audio saved to: {result['audio_file']}")
            elif result.get("voice_error"):
                print(f"✗ Voice error: {result['voice_error']}")
            
            print("="*50)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
