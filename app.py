import streamlit as st
import pandas as pd
import altair as alt
import joblib

# Load model
pipe_lr = joblib.load(open("C:/Users/ajayj/OneDrive/Desktop/Emotive Text Analytics/Training/text_emotion.pkl", "rb"))

# Emojis for emotions
emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔",
                       "sadness": "😔", "shame": "😳", "surprise": "😮"}

# Predict emotion
def predict_emotions(docx):
    return pipe_lr.predict([docx])[0]

# Predict probability
def get_prediction_proba(docx):
    return pipe_lr.predict_proba([docx])

def main():
    st.title("Emotive Text Analytics")
    st.subheader("A project focused on analyzing written text to understand and interpret emotions expressed within it.")

    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        col1, col2 = st.columns(2)
        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)
            st.success("Prediction")
            emoji_icon = emotions_emoji_dict[prediction]
            st.write(f"{prediction}: {emoji_icon}")
            st.write(f"Confidence: {max(probability[0]):.2f}")

        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_).T.reset_index()
            proba_df.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df).mark_bar().encode(x='emotions', y='probability', color='emotions')
            st.altair_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
