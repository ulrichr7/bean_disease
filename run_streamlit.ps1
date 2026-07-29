$port = if ($env:PORT) { $env:PORT } else { "8501" }
streamlit run app.py --server.address 0.0.0.0 --server.port $port
