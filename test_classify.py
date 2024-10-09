from lib.classify import classify_text

# Continuous user input and classification
while True:
    user_input = input("Enter text to classify (or 'quit' to exit): ")
    if user_input.lower() == 'quit':
        break
    predicted_category = classify_text(user_input)
    print(f"Predicted category: {predicted_category}")

print("Thank you for using the classifier!")
