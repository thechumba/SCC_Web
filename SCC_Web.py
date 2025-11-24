import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import streamlit as st
import traceback
import io
from sqlalchemy import create_engine, text
import plotly.express as px
import plotly.graph_objects as go


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


def get_database_connection():
    """Get database connection string from Streamlit secrets."""
    try:
        return st.secrets["database"]["connection_string"]
    except:
        return None


def initialize_database(conn_string):
    """Create necessary tables if they don't exist."""
    try:
        engine = create_engine(conn_string)

        # Create processed_data table
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS processed_data (
                    id SERIAL PRIMARY KEY,
                    processing_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    service_date DATE,
                    entry_date DATE,
                    provider_first_name TEXT,
                    provider_last_name TEXT,
                    provider_name TEXT,
                    charge_code TEXT,
                    time_to_note NUMERIC,
                    date_calculation_issue BOOLEAN,
                    upload_batch_id TEXT
                )
            """))

            # Create summary table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS provider_summary (
                    id SERIAL PRIMARY KEY,
                    processing_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    provider_name TEXT,
                    count INTEGER,
                    mean NUMERIC,
                    median NUMERIC,
                    min NUMERIC,
                    max NUMERIC,
                    std_dev NUMERIC,
                    upload_batch_id TEXT
                )
            """))

            conn.commit()

        return True
    except Exception as e:
        st.error(f"Database initialization error: {str(e)}")
        return False


def save_to_database(df, summary_df, conn_string, batch_id):
    """Save processed data and summary to PostgreSQL database."""
    try:
        engine = create_engine(conn_string)

        # Prepare data for database
        df_to_save = df.copy()

        # Add batch ID and processing date
        df_to_save['upload_batch_id'] = batch_id
        df_to_save['processing_date'] = datetime.now()

        # Convert date columns to proper format
        for col in ['service_date', 'entry_date']:
            if col.title().replace('_', ' ') in df_to_save.columns:
                orig_col = col.title().replace('_', ' ')
                df_to_save[col] = pd.to_datetime(df_to_save[orig_col]).dt.date
                df_to_save = df_to_save.drop(columns=[orig_col])

        # Rename columns to match database schema (lowercase with underscores)
        column_mapping = {
            'Provider First Name': 'provider_first_name',
            'Provider Last Name': 'provider_last_name',
            'Provider Name': 'provider_name',
            'Charge code': 'charge_code',
            'Time to Note': 'time_to_note',
            'Date_Calculation_Issue': 'date_calculation_issue'
        }

        df_to_save = df_to_save.rename(columns=column_mapping)

        # Select only columns that exist in the database
        db_columns = ['processing_date', 'service_date', 'entry_date',
                      'provider_first_name', 'provider_last_name', 'provider_name',
                      'charge_code', 'time_to_note', 'date_calculation_issue', 'upload_batch_id']

        df_to_save = df_to_save[[col for col in db_columns if col in df_to_save.columns]]

        # Save to database
        df_to_save.to_sql('processed_data', engine, if_exists='append', index=False)

        # Save summary
        summary_to_save = summary_df.copy()
        summary_to_save['upload_batch_id'] = batch_id
        summary_to_save['processing_date'] = datetime.now()

        summary_column_mapping = {
            'Provider Name': 'provider_name',
            'Count': 'count',
            'Mean': 'mean',
            'Median': 'median',
            'Min': 'min',
            'Max': 'max',
            'Std Dev': 'std_dev'
        }

        summary_to_save = summary_to_save.rename(columns=summary_column_mapping)
        summary_to_save.to_sql('provider_summary', engine, if_exists='append', index=False)

        return True
    except Exception as e:
        st.error(f"Database save error: {str(e)}")
        st.error(traceback.format_exc())
        return False


def get_historical_data(conn_string, start_date=None, end_date=None):
    """Retrieve historical data from database."""
    try:
        engine = create_engine(conn_string)

        query = """
            SELECT 
                DATE_TRUNC('month', processing_date) as month,
                provider_name,
                AVG(mean) as avg_mean_time,
                AVG(median) as avg_median_time,
                SUM(count) as total_sessions,
                AVG(std_dev) as avg_std_dev
            FROM provider_summary
        """

        conditions = []
        params = {}

        if start_date:
            conditions.append("processing_date >= :start_date")
            params['start_date'] = start_date

        if end_date:
            conditions.append("processing_date <= :end_date")
            params['end_date'] = end_date

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += """
            GROUP BY DATE_TRUNC('month', processing_date), provider_name
            ORDER BY month DESC, provider_name
        """

        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)

        return df
    except Exception as e:
        st.error(f"Error retrieving historical data: {str(e)}")
        return None


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

    # Check for database connection
    conn_string = get_database_connection()

    if conn_string:
        st.success("✅ Database connected")
        initialize_database(conn_string)
    else:
        st.warning("⚠️ No database configured - data will not be saved permanently")
        with st.expander("🔧 How to set up database"):
            st.markdown("""
            1. Create a free Supabase account at https://supabase.com
            2. Create a new project
            3. Get your connection string from Settings → Database
            4. Add it to Streamlit secrets (Settings → Secrets in Streamlit Cloud)

            Format:
```toml
            [database]
            connection_string = "postgresql://postgres:password@host:port/database"
```
            """)

    # Initialize holidays in session state
    if 'holidays' not in st.session_state:
        st.session_state.holidays = setup_holidays()

    # Create tabs
    tab1, tab2 = st.tabs(["📤 Process New Data", "📈 Historical Reports"])

    with tab1:
        st.markdown("""
        Process Excel files containing provider data to calculate **time-to-note metrics**.

        **Note:** Patient identifying information (names) will be removed from displayed outputs for privacy compliance.

        Business days calculation excludes weekends and US Federal holidays.
        """)

        # Sidebar for information
        with st.sidebar:
            st.header("ℹ️ Information")
            st.info(f"""
            **Holidays Configured:** {len(st.session_state.holidays)}

            **Excluded:**
            - Weekends (Sat/Sun)
            - US Federal Holidays

            **Privacy:**
            - Patient names removed from display
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
                "Database name (for download)",
                value="provider_data.db",
                help="SQLite database filename for download"
            )

        # Process button
        if uploaded_file is not None:
            st.success(f"✓ File loaded: {uploaded_file.name}")

            if st.button("🚀 Process Data", type="primary", use_container_width=True):
                with st.spinner("Processing data..."):
                    # Generate batch ID
                    batch_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_file.name}"

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
                        # Save to database if connected
                        if conn_string:
                            with st.spinner("Saving to database..."):
                                if save_to_database(df, summary_df, conn_string, batch_id):
                                    st.success("✅ Data saved to database!")
                                else:
                                    st.warning("⚠️ Processing successful, but failed to save to database")

                        # Remove patient identifying info for display
                        df_display = df.drop(columns=['Patient Last Name', 'Patient First Name', 'Patient Name'],
                                             errors='ignore')

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
                                df_display.to_excel(writer, sheet_name='Processed Data', index=False)
                                summary_df.to_excel(writer, sheet_name='Summary', index=False)

                            st.download_button(
                                label="📥 Download Excel",
                                data=output.getvalue(),
                                file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_processed.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )

                        with col2:
                            # CSV download (processed data)
                            csv_data = df_display.to_csv(index=False)
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
                            st.dataframe(df_display, use_container_width=True)

                    else:
                        st.error("❌ Processing failed. Check the log for details.")

    with tab2:
        st.header("📈 Historical Trend Analysis")

        if not conn_string:
            st.warning("⚠️ Database not configured. Historical reports require database connection.")
            st.info("Set up a database to track provider performance over time.")
        else:
            # Date range selector
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=180))
            with col2:
                end_date = st.date_input("End Date", value=datetime.now())

            if st.button("📊 Generate Report", type="primary"):
                with st.spinner("Loading historical data..."):
                    historical_data = get_historical_data(conn_string, start_date, end_date)

                    if historical_data is not None and not historical_data.empty:
                        st.success(f"✅ Loaded data for {len(historical_data)} provider-months")

                        # Display data table
                        st.subheader("📋 Monthly Averages by Provider")

                        # Format the month column
                        historical_data['month'] = pd.to_datetime(historical_data['month']).dt.strftime('%Y-%m')

                        # Display table
                        st.dataframe(
                            historical_data.sort_values(['month', 'provider_name'], ascending=[False, True]),
                            use_container_width=True,
                            hide_index=True
                        )

                        # Line chart showing trends
                        st.subheader("📈 Mean Time to Note Trends")

                        fig = px.line(
                            historical_data,
                            x='month',
                            y='avg_mean_time',
                            color='provider_name',
                            title='Provider Performance Over Time',
                            labels={
                                'month': 'Month',
                                'avg_mean_time': 'Average Mean Time to Note (days)',
                                'provider_name': 'Provider'
                            }
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Bar chart for latest month
                        st.subheader("📊 Latest Month Comparison")
                        latest_month = historical_data['month'].max()
                        latest_data = historical_data[historical_data['month'] == latest_month]

                        fig2 = px.bar(
                            latest_data.sort_values('avg_mean_time'),
                            x='provider_name',
                            y='avg_mean_time',
                            title=f'Provider Performance - {latest_month}',
                            labels={
                                'provider_name': 'Provider',
                                'avg_mean_time': 'Mean Time to Note (days)'
                            }
                        )

                        st.plotly_chart(fig2, use_container_width=True)

                        # Download historical data
                        st.subheader("💾 Download Historical Data")
                        csv_historical = historical_data.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Historical Report (CSV)",
                            data=csv_historical,
                            file_name=f"historical_report_{start_date}_{end_date}.csv",
                            mime="text/csv"
                        )

                    else:
                        st.info("📭 No historical data found for the selected date range.")


if __name__ == "__main__":
    main()