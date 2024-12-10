import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
        
counter = 0

def get_response(user_input):
    # Step 1: Predict the tag (from your model or logic)
    tag = intent(user_input)[0]  # This could be from your ML model
    
    # Step 2: Check if the predicted tag is valid
    if not tag in [intent['tag'] for intent in intents]:
        return "I'm not sure how to respond to that. Could you rephrase?"
    
    # Step 3: If tag is valid, retrieve response
    for intent in intents:
        if intent in intent["patterns"]:
            user_input = input(" ")
            response = get_response(user_input)
            return random.choice(intent['responses'])  # Choose a random response for variety


def main():
    global counter
    st.title("I am here to assist you!")

    # Create a sidebar menu with options
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu
    if choice == "Home":
        st.write("Welcome to the chatbot. Please enter your query and press Enter to start the conversation.")
        st.write("This chatbot helps you, if you are feeling lonely or you want to chat, Here I'm with you")
        # Check if the chat_log.csv file exists, and if not, create it with column names
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:

            # Convert the user input to a string
            user_input_str = str(user_input)

            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

            # Save the user input and chatbot response to the chat_log.csv file
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    # Conversation History Menu
    elif choice == "Conversation History":
        # Display the conversation history in a collapsible expander
        st.header("Conversation History")
        # with st.beta_expander("Click to see Conversation History"):
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")

    elif choice == "About":
        st.write("The goal of this project is to create a chatbot that can understand and respond to user input based on intents. The chatbot is built using Natural Language Processing (NLP) library and Logistic Regression, to extract the intents and entities from user input. The chatbot is built using Streamlit, a Python library for building interactive web applications.")

        st.subheader("Project Overview:")

        st.write("""
        The project is divided into two parts:
        1. NLP techniques and Logistic Regression algorithm is used to train the chatbot on labeled intents and entities.
        2. For building the Chatbot interface, Streamlit web framework is used to build a web-based chatbot interface. The interface allows users to input text and receive responses from the chatbot.
        """)

        st.subheader("Dataset:")

        st.write("""
        The dataset used in this project is a collection of labelled intents and entities. The data is stored in a list.
        - Intents: The intent of the user input (e.g. "greeting", "budget", "about")
        - Entities: The entities extracted from user input (e.g. "Hi", "How do I create a budget?", "What is your purpose?")
        - Text: The user input text.
        """)

        st.subheader("Streamlit Chatbot Interface:")

        st.write("The chatbot interface is built using Streamlit. The interface includes a text input box for users to input their text and a chat window to display the chatbot's responses. The interface uses the trained model to generate responses to user input.")

        st.subheader("Conclusion:")

        st.write("In this project, a chatbot is built that can understand and respond to user input based on intents. The chatbot was trained using NLP and Logistic Regression, and the interface was built using Streamlit. This project can be extended by adding more data, using more sophisticated NLP techniques, deep learning algorithms.")

if __name__ == '__main__':
    main() 
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "e2fd8a30",
   "metadata": {},
   "source": [
    "# Chatbot"
   ]
  },
  {
   "cell_type": "raw",
   "id": "e909286a",
   "metadata": {},
   "source": [
    "Now let’s start with creating an end-to-end chatbot using Python. I’ll start this task by importing the necessary Python libraries for this task:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "d1f646be",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package punkt to\n",
      "[nltk_data]     C:\\Users\\Sanket\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package punkt is already up-to-date!\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import os\n",
    "import json",
    "import datetime",
    "import csv\n",
    "import nltk\n",
    "import ssl\n",
    "import streamlit as st\n",
    "import random\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "\n",
    "ssl._create_default_https_context = ssl._create_unverified_context\n",
    "nltk.data.path.append(os.path.abspath(\"nltk_data\"))\n",
    "nltk.download('punkt')"
   ]
  },
  {
   "cell_type": "raw",
   "id": "2c1b4e81",
   "metadata": {},
   "source": [
    "Now let’s define some intents of the chatbot. You can add more intents to make the chatbot more helpful and more functional:"
   ]
  },
 ],
   "cell_type": "code",
   "execution_count": 2,
   "id": "d007afc7",
   "metadata": {},
   "outputs": [],
   "source": [
    "intents = [\n",
    "    {\n",
    "        \"tag\": \"greeting\",\n",
    "        \"patterns\": [\"Hi\", \"Hello\", \"Hey\"],\n",
    "        \"responses\": [\"Hi there\", \"Hello\", \"Hey\"]\n",
    "    },\n",
    "    {\n",
    "        \"tag\": \"asking about the person\",\n",
    "        \"patterns\": [\"How are you\", \"What's up\", \"How is your health\"],\n",
    "        \"responses\": [\"I'm fine, thankyou\", \"Nothing much\", \"I'm good what about you\"]\n",
    "    },\n",
    "    {\n",
    "        \"tag\": \"goodbye\",\n",
    "        \"patterns\": [\"Bye\", \"See you later\", \"Goodbye\", \"Take care\"],\n",
    "        \"responses\": [\"Goodbye\", \"See you later\", \"Take care\"]\n",
    "    },\n",
    "    {\n",
    "        \"tag\": \"thanks\",\n",
    "        \"patterns\": [\"Thank you\", \"Thanks\", \"Thanks a lot\", \"I appreciate it\"],\n",
    "        \"responses\": [\"You're welcome\", \"No problem\", \"Glad I could help\"]\n",
    "    },\n",
    "    {\n",
    "        \"tag\": \"about\",\n",
    "        \"patterns\": [\"What can you do\", \"Who are you\", \"What are you\", \"What is your purpose\"],\n",
    "        \"responses\": [\"I am a chatbot\", \"My purpose is to assist you\", \"I can answer questions and provide assistance\"]\n",
    "    },\n",
    "    {\n",
    "        \"tag\": \"help\",\n",
    "        \"patterns\": [\"Help\", \"I need help\", \"Can you help me\", \"What should I do\"],\n",
    "        \"responses\": [\"Sure, what do you need help with?\", \"I'm here to help. What's the problem?\", \"How can I assist you?\"]\n",
    "    },\n",
    "    {\n",
    "        \"tag\": \"age\",\n",
    "        \"patterns\": [\"How old are you\", \"What's your age\"],\n",
    "        \"responses\": [\"I don't have an age. I'm a chatbot.\", \"I was just born in the digital world.\", \"Age is just a number for me.\"]\n",
    "    },\n",
    "    {\n",
    "        \"tag\": \"Health\",\n",
    "        \"patterns\": [\"I got cold\", \"I have fever\", \"I have headache\", \"I have stomach ache\"]\n",
    "        \"responses\": [\"I'm sorry to hear that! Make sure to rest, stay hydrated, make sure you take tablets and precautions and take care of yourself. Hope you feel better soon!\", \"I'm sorry you're feeling unwell. Make sure to rest, drink plenty of fluids, and take fever-reducing medication if needed. If it doesn't improve, consider seeing a doctor. Get well soon!\", \"Sorry you're feeling this way! Rest, hydrate, and take pain relief if needed. Hope you feel better soon!\", \"Sorry to hear that! Try resting, drink some warm water, and avoid heavy food. If it persists, consider seeing a doctor. Get well soon!\"]\n",
    "    },\n",
    "    {\n",
    "        \"tag\": \"weather\",\n",
    "        \"patterns\": [\"What's the weather like\", \"How's the weather today\"],\n",
    "        \"responses\": [\"I'm sorry, I cannot provide real-time weather information.\", \"You can check the weather on a weather app or website.\"]\n",
    "    },\n",
    "    {\n",
    "        \"tag\": \"Asking situation\",\n",
    "        \"patterns\": [\"I am feeling bored\"],\n",
    "        \"responses\": [\"I understand, it can be tough when you're feeling bored. Maybe trying a new activity or hobby could help pass the time—like reading, listening to music, or even a little walk if you're up for it, if you are still feeling bored, talk to you freinds, family, children, play with pet animals.\"]\n",
    "    },\n",
    "    {\n", 
    "        \"tag\": \"family\",\n",
    "        \"patterns\": [\"How is your family?\", \"Tell me about your family\", \"How are the kids?\", \"How's everyone at home?\"],\n",
    "        \"responses\": [\"How lovely to hear you ask! I hope your family is doing well.\", \"Family is so important, isn't it? How is everyone doing?\", \"I hope your loved ones are all in good health.\"]\n",
    "    },\n",
    "    {\n",
    "        \"tag\": \"memories\",\n",
    "        \"patterns\": [\"Tell me a story from your past, Do you remember when?\", \"Can you share an old memory?\", \"What's a favorite memory of yours?\"],\n",
    "        \"responses\": [\"Ah, memories from the past! Those are always special. What's one of your favorites?\", \"I bet you have many wonderful memories. Would you like to share one?\", \"I would love to hear a memory you cherish.\"]\n",
    "    },\n",
    "    {\n",
    "       \"tag\": \"current events\",\n",
    "       \"patterns\": [\"What’s going on in the world today?\", \"Have you heard about the news?\", \"What’s happening around here?\"],\n",
    "       \"responses\": [\"I try not to follow too much news, it can be overwhelming\", \"I heard there’s been some changes in the community\", \"Things seem to be changing fast, but I mostly keep to myself these days\"]\n",
    "   },\n",
    "   {\n",
    "       \"tag\": \"technology confusion\",\n",
    "       \"patterns\": [\"I don’t understand this technology\", \"How do I use this gadget?\", \"Can you help me with my phone/computer?\"],\n",
    "       \"responses\": [\"Technology can be tricky sometimes, but I can help walk you through it\", \"Let me show you how it works – it's easier than it looks!\", \"It can be a bit confusing, but don't worry, you'll get the hang of it\"],\n",
    "   },\n",
    "   {\n",
    "      \"tag\": \"comfort\",\n",
    "      \"patterns\": [\"I'm feeling a bit lonely\", \"It's hard being on my own\", \"I feel a little down today\"],\n",
    "      \"responses\": [\"I’m sorry you feel that way, I’m here for you\", \"You're not alone, we can talk whenever you need\", \"It's okay to feel like that sometimes, but it will pass\"]\n",
    "   },\n",
    "   {\n",
    "      \"tag\": \"pets\",\n",
    "      \"patterns\": [\"Do you have any pets?\", \"Tell me about your pets\", \"What animals do you like?\"],\n", 
    "      \"responses\": [\"I have a cat that keeps me company, she’s always by my side\", \"I love dogs! They're so loyal and friendly\", \"I used to have a dog, but now I enjoy feeding the birds outside\"]\n",
    "   },\n",
    "   {\n",
    "      \"tag\": \"gifts\",\n",
    "      \"patterns\": [\"What would you like for your birthday?\", \"Do you like receiving gifts?", "What’s the best gift you ever received?\"],\n",
    "      \"responses\": [\"I appreciate thoughtful gifts, but honestly, time spent with family is the best gift\", \"I’ve always liked handmade gifts; they come from the heart\", \"The best gift I received was when my children surprised me with a visit\"]\n",
    "    },\n",
    "    {\n",
    "      \"tag\": \"travel memories\",\n",
    "      \"patterns\": [\"What’s your favorite place you’ve traveled to?\", \"Have you traveled a lot?\", \"Where would you love to visit again?\"],\n",
    "      \"responses\": [\"I loved visiting the countryside, it was so peaceful\", \"I used to travel a lot when I was younger, especially to the beach\", \"I’d love to visit the mountains again, I have such fond memories of it\"]\n",        
    "    },\n",
    "    {\n",
    "      \"tag\": \"routine\",\n",
    "      \"patterns\": [\"Do you have a daily routine?\", \"What’s your typical day like?\", \"How do you start your day?\"],\n",
    "      \"responses\": [\"I like to start my day with a cup of tea and a walk around the garden\", \"I usually spend time reading the newspaper, then take a nap\", \"I keep a simple routine: breakfast, a little exercise, and some quiet time\"]\n",
    "    },\n",
    "    {\n",
    "      \"tag\": \"support\",\n",
    "      \"patterns\": [\"I could use some advice\", \"Can you help me with this?\", \"I’m not sure what to do about...\"],\n",
    "      \"responses\": [\"Of course, I’ll help however I can\", \"Don’t worry, we’ll figure this out together\", \"Let’s talk it through, you don’t have to do this alone\"]\n",
    "    },\n",
    "    {\n",
    "      \"tag\": \"self-care\",\n",
    "      \"patterns\": [\"How do you take care of yourself?\", \"What helps you relax?\", \"What’s your self-care routine?\"],\n",
    "      \"responses\": [\"I enjoy a nice warm bath, it helps me relax\", \"I like to meditate or listen to soothing music\", \"A good book and some tea are my perfect way to unwind\"]\n",
    "    },\n",
    "    {\n",
    "       \"tag\": \"jokes\",\n",
    "       \"patterns\": [\"Tell me a joke", "Make me laugh\", \"Do you know any funny stories?\"],\n",
    "       \"responses\": [\"Why don’t skeletons fight each other? They don’t have the guts!\", \"A woman tells her husband, 'You’ve got to stop acting like a flamingo.' So he put his foot down\", \"I remember a time when I tried to be funny... but it didn’t go as planned!\"]\n",
    "   },\n",
    "   {\n",
    "       \"tag\": \"memory loss\",\n",
    "       \"patterns\": [\"I forget things a lot lately\", \"My memory isn’t what it used to be\", \"I can’t remember some things\"],\n",
    "       \"responses\": [\"It's normal to forget things from time to time, don't be too hard on yourself\", \"We all forget things, it’s just part of life\", \"Try writing things down or setting reminders, it helps me a lot\"]\n",
    "   },\n",
    "   {\n",
    "       \"tag\": \"new experiences\",\n",
    "       \"patterns\": [\"Have you tried anything new lately?\", \"What’s something you’ve always wanted to try?\", \"Are you open to new experiences?\"],\n",
    "       \"responses\": [\"I’ve been trying new recipes lately, it’s a fun challenge\", \"I’ve always wanted to take a painting class\", \"I like to keep my mind open to new things, it keeps life interesting\"]\n",
    "   },\n",
    "   {\n",
    "      \"tag\": \"future thoughts\",\n",
    "      \"patterns\": [\"What do you think about the future?\", \"What are your hopes for the future?\", \"What are you looking forward to?\"],\n",
    "      \"responses\": [\"I’m hopeful, especially for my family’s well-being\", \"I look forward to spending more time with my loved ones\", \"I hope to see the world continue to become a kinder place\"]\n",
    "    },\n",
    "    {\n",
    "     \"tag\": \"mindfulness\",\n",
    "     \"patterns\": [\"How do you stay calm?\", \"What helps you be present in the moment?\", \"Do you practice mindfulness?\"],\n",
    "     \"responses\": [\"I take deep breaths and try to appreciate the little things\", \"Sometimes, just sitting in silence helps me clear my mind\", \"I’ve found that focusing on one thing at a time brings peace\"]\n",
    "   },\n",
    "   {\n",
    "    \"tag\": \"independence\",\n",
    "    \"patterns\": [\"Do you like doing things on your own?\", \"Do you need help with anything?\", \"How do you manage things by yourself?\"],\n",
    "    \"responses\": [\"I like doing things on my own when I can, but I know when to ask for help\", \"I try to stay independent, but family is always there when I need them\", \"I manage fine, but it’s nice to have help sometimes\"]\n",
    "   },\n",
    "   {\n",
    "    \"tag\": \"exercise\",\n",
    "    \"patterns\": [\"Do you exercise?\", \"How do you stay active?\", \"What’s your exercise routine like?\"],\n",
    "    \"responses\": [\"I try to walk every day, it keeps me feeling good\", \"I do some light stretches and take short walks around the house\", \"I’ve been doing some gentle yoga; it’s been great for my joints\"]\n",
    "   },\n",
    "  {\n",
    "    \"tag\": \"spirituality\",\n",
    "    \"patterns\": [\"Do you believe in something higher?\", \"What are your spiritual beliefs?\", \"How do you find peace?\"],\n",
    "    \"responses\": [\"I find peace in my faith, it has always been a source of strength\", \"I believe in kindness and helping others\", \"Faith and family are my foundation, they help me stay grounded\"]\n",
    "  },\n",
    "  {\n",
    "    \"tag\": \"current hobbies\",\n",
    "    \"patterns\": [\"What are you into these days?\", \"What hobbies do you enjoy now?\", \"What keeps you busy these days?\"],\n",
    "    \"responses\": [\"I’ve been knitting a lot lately, it’s very calming\", \"I enjoy painting, it’s a great way to express myself\", \"I spend a lot of time reading and enjoying nature\"]\n",
    "   },\n", 
    "   {\n",
    "    \"tag\": \"food\",\n",
    "    \"patterns\": [\"What’s your favorite food?\", \"Do you enjoy cooking?\", \"What’s your go-to comfort food?\"],\n",
    "    \"responses\": [\"I love a good homemade stew, it’s so comforting\", \"I enjoy cooking when I have time, especially baking\", \"My favorite comfort food is mashed potatoes with gravy\"]\n",
    "   },\n",
    "   {\n",
    "     \"tag\": \"gardening\",\n",
    "     \"patterns\": [\"Do you enjoy gardening?\", \"What do you grow in your garden?\", \"How does your garden look?\"],\n",
    "     \"responses\": [\"I love gardening, it’s so peaceful and rewarding\", \"I grow roses, tomatoes, and a few herbs in my garden\", \"My garden is my little sanctuary; it’s full of flowers and plants\"]\n",
    "   },\n",
    "   {\n",
    "     \"tag\": \"satisfaction\",\n",
    "     \"patterns\": [\"What makes you feel proud?\", \"What achievements are you most proud of?\", \"What do you feel most satisfied with?\"],\n",
    "     \"responses\": [\"I’m proud of raising my family, it’s the greatest accomplishment\", \"I’m satisfied knowing I’ve lived a full, honest life\", \"I feel proud of the things I’ve learned and the people I’ve helped\"]\n",
    "  },\n",
    "  {\n",
    "     \"tag\": \"nostalgia\",\n",
    "     \"patterns\": [\"Do you miss the old days?\", \"What was life like back in the day?\", \"Do you ever feel nostalgic?\"],\n",
    "     \"responses\": [\"I do miss certain things, like the slower pace of life\", \"It was a different time, but there was something special about it\", \"Sometimes, I get nostalgic about the way people treated each other back then\"]\n",
    "   },\n",
    "   {\n",
    "      \"tag\": \"community\",\n",
    "      \"patterns\": [\"Do you get involved in your community?\", \"What’s the community like where you live?\", \"Do you know your neighbors well?\"],\n",
    "      \"responses\": [\"Yes, I try to stay involved, it’s nice to know the people around you\", \"Our community is quite close-knit, we help each other out\", \"I know my neighbors well; we check in on each other regularly\"]\n",
    "   },\n",
    "   {\n",
    "      \"tag\": \"fashion\",\n",
    "      \"patterns\": [\"Do you like fashion?\", \"Do you care about what you wear?\", \"What kind of clothes do you like?\"],\n",
    "      \"responses\": [\"I like dressing comfortably, but I do enjoy wearing nice things for special occasions\", \"I’m more about practicality, but I do love a good pair of shoes\", \"I still enjoy getting dressed up for family gatherings or outings\"]\n",
    "   },\n",
    "   {\n",
    "     \"tag\": \"volunteering\",\n",
    "     \"patterns\": [\"Do you volunteer anywhere?\", \"Have you ever done volunteer work?\", \"What’s it like to volunteer?\"],\n",
    "     \"responses\": [\"I’ve volunteered at the local food bank, it’s a very rewarding experience\", \"Yes, I used to help at the senior center; it was a great way to give back", "Volunteering is a wonderful way to stay active and feel connected to others\"]\n",
    "   },\n",
    "   {\n",
    "     \"tag\": \"work\",\n",
    "     \"patterns\": [\"Do you miss working?\", \"What did you do for work?\", \"How did you spend your working years?\"],\n",
    "     \"responses\": [\"I don’t miss it much, but I enjoyed my career when I was working\", \"I worked as a teacher for many years, it was so fulfilling\", \"I spent my career as a nurse, helping people – it was a meaningful job\"]\n",
    "   },\n",
    "   {\n",
    "     \"tag\": \"humor\",\n",
    "     \"patterns\": [\"Do you like jokes?\", \"Do you laugh a lot?\", \"What’s the funniest thing you’ve ever seen?\"],\n",
    "     \"responses\": [\"I love a good laugh, it keeps the spirit young\", \"Humor is important, it’s good for the soul\", \"I remember a time when my dog stole my lunch… I couldn't stop laughing\"]\n",
    "   },\n",
    "   {\n",
    "      \"tag\": \"travel dreams\",\n",
    "      \"patterns\": [\"Where would you love to travel to?\", \"Do you have any travel dreams?\", \"Is there somewhere you still want to visit?\"],\n",
    "      \"responses\": [\"I’ve always wanted to visit Paris, it’s on my bucket list\", \"I dream of going to the Swiss Alps someday, just for the beauty of it\", \"I would love to visit the beaches in Hawaii, I’ve heard they’re stunning\"]\n",
    "   },\n",
    "   {\n",
    "     \"tag\": \"new hobbies\",\n",
    "     \"patterns\": [\"Have you picked up any new hobbies?\", \"What new activity are you trying?\", \"Do you like trying new things?\"],\n",
    "     \"responses\": [\"I’ve started doing puzzles, they’re a good way to keep my mind sharp\", \"I’ve been learning how to paint, it’s relaxing and fun\", \"I’ve been experimenting with different knitting patterns lately\"]\n",
    "    },\n",
    "    {\n",
    "      \"tag\": \"life lessons\",\n",
    "      \"patterns\": [\"What’s the biggest life lesson you’ve learned?\", \"What advice would you give younger people?\", \"What’s something important you’ve learned over the years?\"],\n",
    "      \"responses\": [\"The biggest lesson I’ve learned is that patience is key\", \"My advice would be to cherish your family, they are what matter most\", \"I’ve learned to let go of the little things and focus on what truly matters\"]\n",
    "   },\n",
    "   {\n",
    "      \"tag\": \"birthday memories\",\n",
    "      \"patterns\": [\"What’s your favorite birthday memory?\", \"Do you like celebrating birthdays?\", \"What was your best birthday party?\"],\n",
    "      \"responses\": [\"My best birthday memory was when my kids threw me a surprise party\", \"I enjoy celebrating birthdays, it’s a time to feel special\", \"One of my best birthday memories was when I turned 50, surrounded by friends and family\"]\n",
    "   },\n",
    "   {\n",
    "      \"tag\": \"long-term relationships\",\n",
    "      \"patterns\": [\"What’s the secret to a long-lasting relationship?\", \"How did you stay together for so long?\", \"What advice do you have for couples?\"],\n",
    "      \"responses\": [\"Communication, respect, and a lot of patience are key\", \"The secret is making time for each other and appreciating the little things\", \"Don’t go to bed angry, and always have each other’s back\"]\n",
    "   },\n",
    "   {\n",
    "      \"tag\": \"comfort food\",\n",
    "     \"patterns\": [\"What’s your favorite comfort food?\", \"What food reminds you of home?\", \"What do you cook when you’re feeling down?\"],\n",
    "     \"responses\": [\"I love a good pot roast, it brings back memories of family dinners", "Macaroni and cheese always reminds me of my childhood", "When I need comfort food, I make a big bowl of soup and fresh bread\"]\n",
    "   },\n",
    "   {\n",
    "     \"tag\": \"end of life reflections\",\n",
    "     \"patterns\": [\"Have you thought about what you want when the time comes?\", \"How do you feel about the end of life?\", \"What would you like to be remembered for?\"],\n",
    "     \"responses\": [\"I’ve thought about it, and I just hope to be remembered with love\", \"I’m at peace with it; I just want to know my family will be okay\", \"I would like to be remembered as someone who was kind and always there for others\"]\n",
    "   },\n",
    "   {\n",
    "      \"tag\": \"celebration\",\n",
    "      \"patterns\": [\"What’s your favorite holiday?\", \"How do you like to celebrate holidays?\", \"Do you enjoy family gatherings?\"],\n",
    "      \"responses\": [\"I love Christmas, it’s a time to be with family and reflect\", \"I enjoy quiet celebrations with my family, it’s the best time of the year\", \"Thanksgiving is my favorite, I love the food and the togetherness\"]\n",
    "   },\n",
    "   {\n",
    "     \"tag\": \"technology updates\",\n",
    "     \"patterns\": [\"Have you heard about the new technology?\", \"What do you think of all these new gadgets?\", \"What’s the latest tech news?\"],\n",
    "     \"responses\": [\"I’ve heard a little, but I’m still getting used to some of it\", \"All these new gadgets can be overwhelming, but some are quite handy\", \"Technology changes fast, but I try to keep up with the essentials\"]\n",
    "   },\n",
    "   {\n",
    "      \",tag\": \"good memories\",\n",
    "      \"patterns\": [\"What’s your fondest memory?\", \"Do you have a favorite memory?\", \"What’s the best memory you have?\"],\n",
    "      \"responses\": [\"My fondest memory is when my children were young, they were always so curious\", \"I have so many good memories, it’s hard to choose, but family moments are the best\", \"One of my favorite memories is from a family trip we took years ago, it was magical\"]\n",
    "   },\n",
    "    {\n",
    "        \"tag\": \"budget\",\n",
    "        \"patterns\": [\"How can I make a budget\", \"What's a good budgeting strategy\", \"How do I create a budget\"],\n",
    "        \"responses\": [\"To make a budget, start by tracking your income and expenses. Then, allocate your income towards essential expenses like rent, food, and bills. Next, allocate some of your income towards savings and debt repayment. Finally, allocate the remainder of your income towards discretionary expenses like entertainment and hobbies.\", \"A good budgeting strategy is to use the 50/30/20 rule. This means allocating 50% of your income towards essential expenses, 30% towards discretionary expenses, and 20% towards savings and debt repayment.\", \"To create a budget, start by setting financial goals for yourself. Then, track your income and expenses for a few months to get a sense of where your money is going. Next, create a budget by allocating your income towards essential expenses, savings and debt repayment, and discretionary expenses.\"]\n",
    "    },\n",
    "    {\n",
    "        \"tag\": \"credit_score\",\n",
    "        \"patterns\": [\"What is a credit score\", \"How do I check my credit score\", \"How can I improve my credit score\"],\n",
    "        \"responses\": [\"A credit score is a number that represents your creditworthiness. It is based on your credit history and is used by lenders to determine whether or not to lend you money. The higher your credit score, the more likely you are to be approved for credit.\", \"You can check your credit score for free on several websites such as Credit Karma and Credit Sesame.\"]\n",
    "    }\n",
    ],
    },  {
   "cell_type": "raw",
   "id": "012bf411",
   "metadata": {},
   "source": [
    "Now let’s prepare the intents and train a Machine Learning model for the chatbot:"
   ],
  },{
   "cell_type": "code",
   "execution_count": 3,
   "id": "96d66800",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LogisticRegression(max_iter=10000, random_state=0)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LogisticRegression</label><div class=\"sk-toggleable__content\"><pre>LogisticRegression(max_iter=10000, random_state=0)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "LogisticRegression(max_iter=10000, random_state=0)"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Create the vectorizer and classifier\n",
    "vectorizer = TfidfVectorizer()\n",
    "clf = LogisticRegression(random_state=0, max_iter=10000)\n",
    "\n",
    "# Preprocess the data\n",
    "tags = []\n",
    "patterns = []\n",
    "for intent in intents:\n",
    "    for pattern in intent['patterns']:\n",
    "        tags.append(intent['tag'])\n",
    "        patterns.append(pattern)\n",
    "\n",
    "# training the model\n",
    "x = vectorizer.fit_transform(patterns)\n",
    "y = tags\n",
    "clf.fit(x, y)"
   ]
   },{
   "cell_type": "raw",
   "id": "0639b5f7",
   "metadata": {},
   "source": [
    "Now let’s write a Python function to chat with the chatbot:"
   ]
  },{
   "cell_type": "code",
   "execution_count": 4,
   "id": "563685b4",
   "metadata": {},
   "outputs": [],
   "source": [
    "def chatbot(input_text):\n",
    "    input_text = vectorizer.transform([input_text])\n",
    "    tag = clf.predict(input_text)[0]\n",
    "    for intent in intents:\n",
    "        if intent['tag'] == tag:\n",
    "            response = random.choice(intent['responses'])\n",
    "            return response"
   ]
  }, {
   "cell_type": "raw",
   "id": "471bc8e5",
   "metadata": {},
   "source": [
    "Till now, we have created the chatbot. After running the code, you can interact with the chatbot in the terminal itself. To turn this chatbot into an end-to-end chatbot, we need to deploy it to interact with the chatbot using a user interface. To deploy the chatbot, I will use the streamlit library in Python, which provides amazing features to create a user interface for a Machine Learning application in just a few lines of code."
   ]
  },{
   "cell_type": "raw",
   "id": "196a1376",
   "metadata": {},
   "source": [
    "So, here’s how we can deploy the chatbot using Python:"
   ]
  },  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "1b0080bf",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2023-03-31 17:08:14.216 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\Sanket\\anaconda3\\envs\\nlp\\lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "counter = 0\n",
    "\n",
    "def main():\n",
    "    global counter\n",
    "    st.title(\"Chatbot\")\n",
    "    st.write(\"Welcome to the chatbot. Please type a message and press Enter to start the conversation.\")\n",
    "\n",
    "    counter += 1\n",
    "    user_input = st.text_input(\"You:\", key=f\"user_input_{counter}\")\n",
    "\n",
    "    if user_input:\n",
    "        response = chatbot(user_input)\n",
    "        st.text_area(\"Chatbot:\", value=response, height=100, max_chars=None, key=f\"chatbot_response_{counter}\")\n",
    "\n",
    "        if response.lower() in ['goodbye', 'bye']:\n",
    "            st.write(\"Thank you for chatting with me. Have a great day!\")\n",
    "            st.stop()\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    main()"
   ],
  },{
   "cell_type": "code",
   "execution_count": None, 
   "id": "7124a3ce",
   "metadata": {},
   "outputs": [],
   "source": []
  },{
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.0",
  }
 }
 },{
 "nbformat": 4,
 "nbformat_minor": 5,
 }
