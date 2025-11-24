import pandas as pd
import numpy as np
import os
import sqlite3
from datetime import datetime, timedelta
import streamlit as st
import traceback
import io
from pathlib import Path


def is_weekend(date):
    """Check if a date is a weekend (Saturday=5 or Sunday=6)."""
    return date.weekday() >= 5


def is_holiday(date, holidays):
    """Check if a date is in the holidays list."""
    if holidays is None:
        return False
    return date.strftime('%Y-%m-%d') in holidays


def business_days_between(start_date, end_date, holidays=None):
    """
    Calculate business days between two dates, excluding weekends and holidays.

    Args:
        start_date: Start date as datetime.date object
        end_date: End date as datetime.date object
        holidays: List of holiday dates as strings in format 'YYYY-MM-DD'

    Returns:
        int: Number of business days between the dates (0 if same day)
    """
    if holidays is None:
        holidays = []

    # Ensure we're working with date objects (no time component)
    start_date = start_date.date() if hasattr(start_date, 'date') else start_date
    end_date = end_date.date() if hasattr(end_date, 'date') else end_date

    # If dates are the same, return 0 (same day)
    if start_date == end_date:
        return 0

    # If end date is before start date, return 0 (error condition)
    if start_date > end_date:
        return 0

    # Count the days, excluding weekends and holidays
    days = 0
    current_date = start_date

    while current_date <= end_date:
        # Skip weekends and holidays
        if not is_weekend(current_date) and not is_holiday(current_date, holidays):
            days += 1
        current_date += timedelta(days=1)

    # If start date was a weekend or holiday and we're not counting it,
    # we don't need to subtract 1 from the count
    if is_weekend(start_date) or is_holiday(start_date, holidays):
        return days
    else:
        # Otherwise, subtract 1 to not count the start date
        return max(0, days - 1)


def setup_holidays():
    """
    Define holidays that will be excluded from business day calculations.
    Modify this list as needed for your specific holidays.
    """
    return [
        # 2023 US Federal Holidays
        '2023-01-01', '2023-01-16', '2023-02-20', '2023-05-29',
        '2023-06-19', '2023-07-04', '2023-09-04', '2023-10-09',
        '2023-11-10', '2023-11-23', '2023-12-25',

        # 2024 US Federal Holidays
        '2024-01-01', '2024-01-15', '2024-02-19', '2024-05-27',
        '2024-06-19', '2024-07-04', '2024-09-02', '2024-10-14',
        '2024-11-11', '2024-11-28', '2024-12-25',

        # 2025 US Federal Holidays
        '2025-01-01', '2025-01-20', '2025-02-17', '2025-05-26',
        '2025-06-19', '2025-07-04', '2025-09-01', '2025-10-13',
        '2025-11-11', '2025-11-27', '2025-12-25',
    ]


def process_provider_data(uploaded_file, holidays_list=None):
    """
    Process provider data to calculate time to note metrics.

    Args:
        uploaded_file: Uploaded file object from Streamlit
        holidays_list: Optional list of holidays as strings in format 'YYYY-MM-DD'

    Returns:
        tuple: (processed_df, summary_df, status_messages)
    """
    status_messages = []

    def log_status(message):
        status_messages.append(message)

    # Ensure holidays_list is properly set
    if holidays_list is None:
        holidays_list = []
        log_status("⚠️ Warning: No holidays list provided. Weekends will still be excluded.")
    else:
        log_status(f"✓ Using {len(holidays_list)} holidays for business day calculations")

    # Predefined list of charge codes to exclude
    excluded_charge_codes = [
        96116, 96121, 96310, "consult", "dwseval", 96653, 96137, 96136,
        96131, 96132, 96130, 96133, "dsps", "dspdx", 96750, 96101, 96800,
        "laweval", "CONSULT", "court", "HARE", "LAWNEURO", 90791, 90792,
        99404, "9940mha", "N / A", "n / a", "N/A", "n/a", 2002,
        "h2014", "h2017", "na", 9991
    ]

    log_status(f"📋 Excluded charge codes: {len(excluded_charge_codes)} codes")

    try:
        # Read the Excel file
        log_status("📖 Reading Excel file...")

        # Read without parsing dates first
        df = pd.read_excel(uploaded_file)

        log_status(f"✓ Found {len(df)} rows")
        log_status(f"✓ Columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")

        # Convert date columns
        date_columns = ['Service Date', 'Entry Date']
        for col in date_columns:
            if col in df.columns:
                log_status(f"🔄 Converting {col} to datetime...")
                original_values = df[col].copy()
                df[col] = pd.to_datetime(df[col], errors='coerce')

                nat_count = df[col].isna().sum()
                if nat_count > 0:
                    log_status(f"⚠️ WARNING: {nat_count} rows have invalid date values in {col}")

        # Filter out excluded charge codes
        charge_col = 'Charge code'
        if charge_col in df.columns:
            total_rows_before = len(df)
            df = df[~df[charge_col].isin(excluded_charge_codes)]
            excluded_rows = total_rows_before - len(df)
            log_status(f"🔍 Excluded {excluded_rows} rows with excluded charge codes")

        # Handle name columns
        patient_first_col = 'Patient First Name'
        patient_last_col = 'Patient Last Name'
        provider_first_col = 'Provider First Name'
        provider_last_col = 'Provider Last Name'

        # Create concatenated name columns (needed for processing)
        df['Patient Name'] = df[patient_last_col] + ', ' + df[patient_first_col]
        df['Provider Name'] = df[provider_last_col] + ', ' + df[provider_first_col]

        # Sort by Provider Name
        df = df.sort_values('Provider Name')

        # Calculate Time to Note
        if 'Service Date' in df.columns and 'Entry Date' in df.columns:
            log_status("📊 Calculating time to note (excluding weekends and holidays)...")

            df['Date_Calculation_Issue'] = False
            time_to_note = []
            problematic_count = 0

            for idx, row in df.iterrows():
                service_date = row['Service Date']
                entry_date = row['Entry Date']

                if pd.notnull(service_date) and pd.notnull(entry_date):
                    try:
                        service_date_only = service_date.date() if hasattr(service_date, 'date') else service_date
                        entry_date_only = entry_date.date() if hasattr(entry_date, 'date') else entry_date

                        if service_date_only == entry_date_only:
                            time_to_note.append(0)
                            continue

                        days_diff = business_days_between(service_date, entry_date, holidays_list)

                        if days_diff < 0:
                            time_to_note.append(np.nan)
                            df.at[idx, 'Date_Calculation_Issue'] = True
                            problematic_count += 1
                        else:
                            time_to_note.append(days_diff)
                    except Exception as e:
                        time_to_note.append(np.nan)
                        df.at[idx, 'Date_Calculation_Issue'] = True
                        problematic_count += 1
                else:
                    time_to_note.append(np.nan)

            df['Time to Note'] = time_to_note

            if problematic_count > 0:
                log_status(f"⚠️ Found {problematic_count} rows with date calculation issues")
            else:
                log_status("✓ All date calculations completed successfully")

        # Create summary statistics
        log_status("📈 Generating summary statistics by provider...")

        summary_df = df.groupby('Provider Name').agg({
            'Time to Note': ['count', 'mean', 'median', 'min', 'max', 'std']
        }).reset_index()

        summary_df.columns = ['Provider Name', 'Count', 'Mean', 'Median', 'Min', 'Max', 'Std Dev']

        for col in ['Mean', 'Median', 'Min', 'Max', 'Std Dev']:
            summary_df[col] = summary_df[col].round(2)

        summary_df = summary_df.sort_values('Provider Name')

        # *** REMOVE PATIENT IDENTIFYING COLUMNS ***
        log_status("🔒 Removing patient identifying information from output...")
        columns_to_remove = ['Patient Last Name', 'Patient First Name', 'Patient Name']

        # Keep track of which columns exist before removing
        existing_columns_to_remove = [col for col in columns_to_remove if col in df.columns]

        if existing_columns_to_remove:
            df = df.drop(columns=existing_columns_to_remove)
            log_status(f"✓ Removed columns: {', '.join(existing_columns_to_remove)}")

        log_status("✅ Processing complete!")

        return df, summary_df, status_messages

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n{traceback.format_exc()}"
        log_status(error_msg)
        return None, None, status_messages


def main():
    # Page configuration
    st.set_page_config(
        page_title="Provider Data Processor",
        page_icon="📊",
        layout="wide"
    )

    # Title and description
    st.title("📊 Provider Data Processor")
    st.markdown("""
    Process Excel files containing provider data to calculate **time-to-note metrics**.

    **Note:** Patient identifying information (names) will be removed from all outputs for privacy compliance.

    Business days calculation excludes weekends and US Federal holidays.
    """)

    # Initialize holidays in session state
    if 'holidays' not in st.session_state:
        st.session_state.holidays = setup_holidays()

    # Sidebar for information
    with st.sidebar:
        st.header("ℹ️ Information")
        st.info(f"""
        **Holidays Configured:** {len(st.session_state.holidays)}

        **Excluded:**
        - Weekends (Sat/Sun)
        - US Federal Holidays

        **Privacy:**
        - Patient names removed from output
        """)

        with st.expander("📋 Required Excel Format"):
            st.markdown("""
            Your Excel file must contain:
            - Service Date
            - Entry Date
            - Provider First/Last Name
            - Patient First/Last Name
            - Charge code
            """)

        with st.expander("🚫 Excluded Charge Codes"):
            st.markdown("""
            Certain charge codes are automatically excluded 
            (consultations, evaluations, etc.)
            """)

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📁 Upload Excel File")
        uploaded_file = st.file_uploader(
            "Choose an Excel file",
            type=['xlsx', 'xls'],
            help="Upload your provider data Excel file"
        )

    with col2:
        st.header("⚙️ Settings")
        db_name = st.text_input(
            "Database name",
            value="provider_data.db",
            help="SQLite database filename"
        )

    # Process button
    if uploaded_file is not None:
        st.success(f"✓ File loaded: {uploaded_file.name}")

        if st.button("🚀 Process Data", type="primary", use_container_width=True):
            with st.spinner("Processing data..."):
                # Process the data
                df, summary_df, status_messages = process_provider_data(
                    uploaded_file,
                    st.session_state.holidays
                )

                # Display status messages
                with st.expander("📋 Processing Log", expanded=True):
                    for msg in status_messages:
                        st.text(msg)

                if df is not None and summary_df is not None:
                    st.success("✅ Processing completed successfully!")

                    # Display summary
                    st.header("📊 Summary Statistics by Provider")
                    st.dataframe(
                        summary_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    # Visualizations
                    st.header("📈 Visualizations")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Mean Time to Note by Provider")
                        chart_data = summary_df.set_index('Provider Name')['Mean']
                        st.bar_chart(chart_data)

                    with col2:
                        st.subheader("Distribution of Time to Note")
                        st.line_chart(df['Time to Note'].value_counts().sort_index())

                    # Download section
                    st.header("💾 Download Results")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        # Excel download
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df.to_excel(writer, sheet_name='Processed Data', index=False)
                            summary_df.to_excel(writer, sheet_name='Summary', index=False)

                        st.download_button(
                            label="📥 Download Excel",
                            data=output.getvalue(),
                            file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_processed.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                    with col2:
                        # CSV download (processed data)
                        csv_data = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV (Data)",
                            data=csv_data,
                            file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_processed.csv",
                            mime="text/csv"
                        )

                    with col3:
                        # CSV download (summary)
                        csv_summary = summary_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV (Summary)",
                            data=csv_summary,
                            file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_summary.csv",
                            mime="text/csv"
                        )

                    with col4:
                        # SQLite database
                        db_output = io.BytesIO()
                        conn = sqlite3.connect(':memory:')

                        df_to_save = df.copy()
                        for col in ['Service Date', 'Entry Date']:
                            if col in df_to_save.columns:
                                df_to_save[col] = df_to_save[col].astype(str)

                        df_to_save.to_sql('processed_data', conn, if_exists='replace', index=False)
                        summary_df.to_sql('summary_by_provider', conn, if_exists='replace', index=False)

                        # Write database to bytes
                        for line in conn.iterdump():
                            db_output.write(f'{line}\n'.encode('utf-8'))
                        conn.close()

                        st.download_button(
                            label="📥 Download Database",
                            data=db_output.getvalue(),
                            file_name=db_name,
                            mime="application/x-sqlite3"
                        )

                    # Show detailed data
                    with st.expander("🔍 View Processed Data"):
                        st.dataframe(df, use_container_width=True)

                else:
                    st.error("❌ Processing failed. Check the log for details.")


if __name__ == "__main__":
    main()