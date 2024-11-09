import streamlit as st
import pandas as pd
import plotly.express as px
import openai
import os

# Streamlit configuration
st.set_page_config(page_title="Fan Engagement Analysis", layout="wide")

# Set up OpenAI API key (Make sure to set your API key in your environment)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your environment.")
else:
    openai.api_key = OPENAI_API_KEY

# Load the processed data for visualization
@st.cache_data
def load_data():
    return pd.read_csv('data/cleaned_fan_data.csv')

# Load the cleaned fan data
data = load_data()

# Sidebar navigation
st.sidebar.title("Navigation")
page_selection = st.sidebar.radio("Go to", ["What-If Analysis Dashboard", "LLM Chatbot Q&A"])

# Page 1: What-If Analysis Dashboard
if page_selection == "What-If Analysis Dashboard":
    st.title("What-If Analysis Dashboard")

    # Summary Metrics Section
    st.markdown("### All-Time Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_fans = len(data)
        st.metric(label="Total Fans", value=total_fans)
    with col2:
        total_ticket_units = data['Lifetime_Ticket_Units'].sum()
        st.metric(label="Total Ticket Units", value=total_ticket_units)
    with col3:
        total_concessions_spend = data['Lifetime_Concessions_Spend'].sum()
        st.metric(label="Total Concessions Spend ($)", value=f"${total_concessions_spend:,.2f}")
    with col4:
        avg_engagement_score = data['Total_Engagement_Score'].mean()
        st.metric(label="Average Engagement Score", value=f"{avg_engagement_score:.2f}")

    # Interactive Scenario Planning
    st.sidebar.header("Scenario Settings")
    discount_percentage = st.sidebar.slider("Ticket Discount Percentage (%)", min_value=0, max_value=50, value=10)
    distance_filter = st.sidebar.slider("Maximum Distance to Arena (miles)", min_value=0, max_value=100, value=50)
    income_level_filter = st.sidebar.selectbox("Filter by Income Level", options=['All', 'Low', 'Medium', 'High'], index=0)
    fan_type_filter = st.sidebar.multiselect("Select Fan Type", options=data['Fan_Type'].unique(), default=data['Fan_Type'].unique())

    # Apply scenario logic to the data
    filtered_data = data[(data['Distance_to_Arena_Miles'] <= distance_filter) & (data['Fan_Type'].isin(fan_type_filter))]
    if income_level_filter != 'All':
        filtered_data = filtered_data[filtered_data['Income_Level'] == {'Low': 1, 'Medium': 2, 'High': 3}[income_level_filter]]
    filtered_data['Adjusted_Ticket_Units'] = filtered_data['Lifetime_Ticket_Units'] * (1 + discount_percentage / 100)

    # Display Charts
    st.markdown("### Selected Duration Metrics")
    metrics = [
        ("Total Ticket Units", "Adjusted_Ticket_Units", '#29b5e8'),
        ("Total Concessions Spend", "Lifetime_Concessions_Spend", '#FF9F36'),
        ("Average Engagement Score", "Total_Engagement_Score", '#D45B90'),
        ("Total Games Attended", "Lifetime_Games_Attended", '#7D44CF')
    ]

    cols = st.columns(4)
    for col, (title, column, color) in zip(cols, metrics):
        total_value = filtered_data[column].sum()
        with col:
            st.metric(label=title, value=total_value)

    # Chart 1: Impact of Ticket Discount on Attendance
    st.markdown("### Impact of Ticket Discount on Attendance by Fan Type")
    fig1 = px.bar(
        filtered_data, x='Fan_Type', y='Adjusted_Ticket_Units', color='Income_Level',
        title=f"Projected Attendance with {discount_percentage}% Ticket Discount",
        labels={
            "Adjusted_Ticket_Units": "Projected Ticket Units Purchased",
            "Fan_Type": "Fan Type"
        }
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Chart 2: Average Lifetime Spend by Fan Type
    st.markdown("### Average Lifetime Spend by Fan Type")
    fig2 = px.bar(
        filtered_data, x='Fan_Type', y='Lifetime_Concessions_Spend', color='Income_Level',
        title="Average Lifetime Concessions Spend by Fan Type",
        labels={
            "Lifetime_Concessions_Spend": "Lifetime Concessions Spend",
            "Fan_Type": "Fan Type"
        }
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Chart 3: Attendance Trend by Income Level
    st.markdown("### Attendance Trend by Income Level")
    fig3 = px.histogram(
        filtered_data, x='Attendance_Trend', color='Income_Level',
        title="Attendance Trend by Income Level",
        labels={
            "Attendance_Trend": "Attendance Trend",
            "Income_Level": "Income Level"
        },
        barmode='group'
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Chart 4: Distance to Arena vs. Ticket Purchases
    st.markdown("### Distance to Arena vs. Ticket Purchases")
    fig4 = px.scatter(
        filtered_data, x='Distance_to_Arena_Miles', y='Lifetime_Ticket_Units', color='Fan_Type',
        title="Distance to Arena vs. Lifetime Ticket Units Purchased",
        labels={
            "Distance_to_Arena_Miles": "Distance to Arena (miles)",
            "Lifetime_Ticket_Units": "Lifetime Ticket Units Purchased"
        }
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Chart 5: Engagement Score Distribution by Fan Type
    st.markdown("### Engagement Score Distribution by Fan Type")
    fig5 = px.box(
        filtered_data, x='Fan_Type', y='Total_Engagement_Score', color='Income_Level',
        title="Engagement Score Distribution by Fan Type",
        labels={
            "Total_Engagement_Score": "Total Engagement Score",
            "Fan_Type": "Fan Type"
        }
    )
    st.plotly_chart(fig5, use_container_width=True)

    # Additional summary
    st.markdown("### Summary of Scenario")
    st.write(f"With a {discount_percentage}% ticket discount, fans living within {distance_filter} miles are expected to increase their ticket purchases.")
    if income_level_filter != 'All':
        st.write(f"This projection is filtered for income level: {income_level_filter}.")
    st.write(f"Fan types considered: {', '.join(fan_type_filter)}")

# Page 2: LLM Chatbot Q&A
elif page_selection == "LLM Chatbot Q&A":
    st.title("LLM Chatbot Q&A")
    st.write("Ask questions about the fan engagement data, and get answers powered by GPT-4.")

    # Input for user question
    user_question = st.text_input("Type your question here:")

    if user_question:
        # Provide contextual information from the dataset
        context = (
            f"The fan engagement dataset contains information about {len(data)} fans. "
            f"The average engagement score is {data['Total_Engagement_Score'].mean():.2f}. "
            f"The total ticket spend is ${data['Lifetime_Ticket_Spend'].sum():,.2f}. "
            f"Recent analysis has shown trends such as attendance trends by income level, "
            f"spending on concessions, and engagement score distributions."
        )

        # Create the prompt including the context and user question
        full_prompt = f"{context} Based on this information, answer the following question: {user_question}"

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=800,  # Allow more detailed suggestions
                temperature=0.7
            )
            answer = response['choices'][0]['message']['content'].strip()
            st.write("**Answer:**")
            st.write(answer)

            # Additional insights based on the question
            if "discount" in user_question.lower():
                additional_info = "You might want to adjust the 'Ticket Discount Percentage' slider in the What-If Analysis Dashboard to visualize how discount changes impact fan engagement."
            elif "income level" in user_question.lower():
                additional_info = "Consider filtering by 'Income Level' in the Scenario Settings to explore how different income levels affect engagement."
            else:
                additional_info = None
            
            if additional_info:
                st.write("**Additional Insight:**")
                st.write(additional_info)

        except Exception as e:
            st.error(f"Error generating response: {e}")
