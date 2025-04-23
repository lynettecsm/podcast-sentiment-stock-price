# NVIDIA Stock Price Prediction Using Podcast Sentiment Analysis

## Project Overview
This project explores how podcast sentiment analysis can enhance traditional stock price prediction models for NVIDIA. By analyzing sentiment from financial podcasts and combining it with technical price features, we created a hybrid FinBERT-LSTM model that outperforms baseline price models by 17.09%.

## Key Features
- Extraction of sentiment from financial podcasts using FinBERT
- NVIDIA-specific rule-based sentiment augmentation
- Integration of sentiment features with stock price technical indicators
- LSTM-based time series prediction model

## Dataset
- **Podcast corpus**: 902 podcast episodes from Seeking Alpha's Wall Street Breakfast, Bloomberg Markets, and CNBC's "Mad Money"
- **Processed corpus**: 9,725 NVIDIA-relevant sentences after filtering
- **Technical data**: NVIDIA stock price history from Yahoo Finance
- **Time period**: Training data prior to October 1, 2024; testing data after

## Methodology
### Data Collection & Processing
1. Downloaded audio files from podcast RSS feeds
2. Transcribed using OpenAI Whisper
3. Preserved metadata (episode title, publication date, timestamp)
4. Cleaned and filtered podcast transcripts for NVIDIA mentions
5. Applied rule-based augmentation for NVIDIA-specific context

### Sentiment Analysis
- Used FinBERT, a financial domain-specific BERT model
- Enhanced sentiment logits using rule-based augmentation
- Applied 7 categories of domain-specific keywords for contextual filtering
- Generated daily sentiment aggregations

### Stock Price Prediction
- Implemented LSTM architecture with two 50-unit layers and dropout
- Combined price features with sentiment features
- Used a rolling 10-day window for prediction
- Evaluated using RMSE and relative performance improvements

## Results
- Best model: Full sentiment feature set (scores, logits, and categorical predictions)
- Overall improvement over baseline price model: 17.09%
- Absolute error reduction: 14.21%
- Superior performance during high volatility periods

## Model Architecture
The model incorporates 16 features including:
- Standard price features (Open, High, Low, Close, Volume)
- Technical indicators (returns, volatility, moving averages)
- Enhanced sentiment features (logits, scores, binary indicators)

## Conclusions
- Podcast sentiment analysis provides valuable signals for stock price prediction
- Enhanced sentiment features offer better performance than basic sentiment scores
- The model shows particular strength during periods of high market volatility
- Future work should focus on feature regularization and model robustness across different market conditions


## Dependencies
- Python 3.9+
- PyTorch
- Transformers (for FinBERT)
- Pandas
- NumPy
- TensorFlow/Keras (for LSTM)
- OpenAI Whisper (for transcription)

## Authors
- Jewel Ornido (1009844)
- Lynette Chia (1006170)

## Institution
Singapore University of Technology and Design  
Computational Data Science  
Data Exchangez

## References
- Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. arXiv preprint arXiv:1908.10063.
- Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. Neurocomputing, 261, 118-138.
- Gu, W., Zhong, Y., Li, S., & Wei, C. (2024). Predicting Stock Prices with FinBERT-LSTM: Integrating News Sentiment Analysis. In Proceedings of the 2024 8th International Conference on Cloud and Big Data Computing (ICCBDC) (pp. 1-7).
- Yang, Z., & Wang, Z. (2024). The Research of NVIDIA Stock Price Prediction Based on LSTM and ARIMA Model. Highlights in Business, Economics and Management, 24, 896-902.
- Chen, H., & Hong, L. (2023). Financial Public Opinion Risk Prediction Model Integrating Knowledge Association and Temporal Transmission. Data Analysis and Knowledge Discovery, 7(11), 1-13.