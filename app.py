import streamlit as st
import pandas as pd
import plotly.express as px
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.title("Data Storyteller")
st.subheader("Upload any CSV and get instant insights")

uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    text_cols = df.select_dtypes(include='str').columns.tolist()

    st.success("File uploaded successfully")
    st.subheader("Preview of your data")
    st.dataframe(df.head(10))

    st.subheader("Basic Stats")
    st.write(df.describe())

    st.subheader("🔍 Filter Your Data")

    with st.expander("Add Filters"):
        filtered_df = df.copy()

        st.write("**Numeric Filters**")
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue
            min_val = float(col_data.min())
            max_val = float(col_data.max())
            if min_val == max_val:
                continue
            unique_vals = sorted(col_data.unique().tolist())
            if len(unique_vals) <= 50:
                selected_vals = st.multiselect(
                    f"{col}",
                    options=unique_vals,
                    default=unique_vals
                )
                if selected_vals:
                    filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]

        st.write("**Category Filters**")
        for col in text_cols:
            unique_vals = df[col].dropna().unique().tolist()
            if len(unique_vals) <= 100:
                selected_vals = st.multiselect(
                    f"{col}",
                    options=unique_vals,
                    default=unique_vals
                )
                if selected_vals:
                    filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]

    st.info(f"✅ Showing {len(filtered_df):,} of {len(df):,} rows after filters")
    st.dataframe(filtered_df.head(10))

    st.subheader("📈 Visualize Your Data")

    df_sample = filtered_df.sample(n=5000, random_state=42) if len(filtered_df) > 5000 else filtered_df

    chart_type = st.selectbox("Select Chart Type", ["Scatter Plot", "Bar Chart", "Box Plot", "Histogram", "Line Chart"])

    if chart_type == "Scatter Plot":
        x_axis = st.selectbox("Select X axis", numeric_cols)
        y_axis = st.selectbox("Select Y axis", numeric_cols)
        color = st.selectbox("Color by (optional)", ["None"] + text_cols)
        color_col = None if color == "None" else color
        chart = px.scatter(df_sample, x=x_axis, y=y_axis, color=color_col, title=f"{x_axis} vs {y_axis}")

    elif chart_type == "Bar Chart":
        x_axis = st.selectbox("Select Category (X axis)", text_cols)
        y_axis = st.selectbox("Select Value (Y axis)", numeric_cols)
        chart = px.bar(df_sample, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")

    elif chart_type == "Box Plot":
        x_axis = st.selectbox("Select Category (X axis)", text_cols)
        y_axis = st.selectbox("Select Value (Y axis)", numeric_cols)
        chart = px.box(df_sample, x=x_axis, y=y_axis, title=f"{y_axis} distribution by {x_axis}")

    elif chart_type == "Histogram":
        x_axis = st.selectbox("Select Column", numeric_cols)
        chart = px.histogram(df_sample, x=x_axis, title=f"Distribution of {x_axis}")

    elif chart_type == "Line Chart":
        x_axis = st.selectbox("Select X axis", numeric_cols)
        y_axis = st.selectbox("Select Y axis", numeric_cols)
        chart = px.line(df_sample, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")

    st.plotly_chart(chart)

    st.subheader("🤖 AI Business Insights")

    user_prompt = st.text_area(
        "What do you want to analyze?",
        placeholder="eg. Find trends, give numbers, count of status 2...",
        height=100
    )

    if st.button("Generate Insights"):
        with st.spinner("Analysing your data...."):

            summary = f"""
            Dataset has {filtered_df.shape[0]} rows and {filtered_df.shape[1]} columns
            Columns: {','.join(filtered_df.columns.tolist())}

            Key Statistics:
            {filtered_df.describe().to_string()}
            """

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data analyst assistant. The user will ask specific questions about their data. Give DIRECT, CONCISE answers only. No introductions, no lengthy explanations. If they ask for a count, give the count. If they ask for a list, give the list. Format results clearly using bullet points or tables. Maximum 10 lines."
                    },
                    {
                        "role": "user",
                        "content": f"User's analysis goal: {user_prompt}\n\nAnalyze this dataset and give me business insights:\n{summary}"
                    }
                ]
            )

            insight = response.choices[0].message.content
            st.markdown(insight)

            st.download_button(
                label="📥 Download Report",
                data=insight,
                file_name="data_insights_report.txt",
                mime="text/plain"
            )