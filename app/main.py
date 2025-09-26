from search_trigger import needs_live_data_openai




def main():
    print("Live Data Classifier (type 'exit' to quit)")

    while True:
        prompt = input("\nEnter your prompt: ").strip()

        if prompt.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        try:
            if needs_live_data_openai(prompt):
                print("Requires live data / web search")
            else:
                print("Can answer from existing knowledge")
        except Exception as e:
            print(f"Error calling model: {e}")

if __name__ == "__main__":
    main()