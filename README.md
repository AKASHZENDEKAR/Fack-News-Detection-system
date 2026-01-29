# 🔍 Fake News Detection System

A machine learning-based web application that analyzes news text and determines whether it's likely to be true or fake news.

## 🌟 Features

- **Smart Text Validation**: Rejects non-news content and short text
- **Length Requirements**: Minimum 50 characters for news analysis
- **News Pattern Detection**: Identifies proper news formatting and indicators
- **Similarity-Based Classification**: Compares input text against 21,417 true news articles
- **Modern Web Interface**: Clean, responsive design with real-time analysis
- **TF-IDF Vectorization**: Advanced text processing for accurate predictions

## 🛠️ Technologies Used

- **Backend**: Python, Flask
- **Machine Learning**: scikit-learn, pandas, NumPy
- **Text Processing**: TF-IDF Vectorization, Cosine Similarity
- **Frontend**: HTML5, CSS3, JavaScript
- **Data**: 21,417 true news articles for training

## 📊 How It Works

1. **Text Preprocessing**: Cleans and validates input text
2. **Length Check**: Ensures minimum 50 characters for news content
3. **Pattern Detection**: Looks for news indicators (Reuters, officials, reported, etc.)
4. **Vectorization**: Converts text to numerical vectors using TF-IDF
5. **Similarity Analysis**: Compares against known true news patterns
6. **Classification**: Returns TRUE (similar to training data) or FAKE (not similar)

## 🚀 Installation & Setup

### Prerequisites
- Python 3.7+
- pip (Python package installer)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/fake-news-detection-system.git
   cd fake-news-detection-system
   ```

2. **Install dependencies**
   ```bash
   pip install flask pandas scikit-learn
   ```

3. **Train the model**
   ```bash
   python train_model.py
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the web interface**
   - Open your browser
   - Go to `http://localhost:8000`

## 📁 Project Structure

```
fake-news-detection-system/
├── app.py                 # Main Flask application
├── train_model.py         # Model training script
├── model.py              # Custom model class
├── templates/
│   └── index.html        # Web interface
├── true.csv              # Dataset (21,417 true news articles)
├── fake_news_model.pkl   # Trained model file
├── requirements.txt      # Dependencies
├── .gitignore           # Git ignore file
└── README.md            # Project documentation
```

## 🧪 Usage Examples

### ✅ Good Examples (Real News)
```
WASHINGTON (Reuters) - The head of a conservative Republican faction in the U.S. Congress, who voted this month for a huge expansion of the national debt to pay for tax cuts, called himself a 'fiscal conservative' on Sunday and urged budget restraint in 2018.
```

### ❌ Bad Examples (Not News)
- "manu is good boy" - Too short, not news format
- "Hello world" - Not news content
- "I like pizza" - Personal statement, not news

### ⚠️ Fake News Example
```
BREAKING: Scientists discover that drinking water causes cancer! A new study shows that 100% of people who drink water will develop cancer within 5 years. Share this important information immediately!
```

## 📊 Model Performance

- **Training Data**: 21,417 true news articles
- **Algorithm**: Similarity-based classification using TF-IDF and Cosine Similarity
- **Threshold**: 0.3 similarity threshold for classification
- **Features**: 5,000 most important words (TF-IDF)

## 🔧 Configuration

### Model Parameters
- **Similarity Threshold**: 0.3 (adjustable in `model.py`)
- **Minimum Text Length**: 50 characters
- **Max Features**: 5,000 TF-IDF features
- **Stop Words**: English stop words removed

### Flask Configuration
- **Host**: 0.0.0.0 (all interfaces)
- **Port**: 8000
- **Debug Mode**: Enabled for development

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**
- GitHub: [https://github.com/AKASHZENDEKAR]
- Email: akashzendekar@gmail.com

## 🙏 Acknowledgments

- Dataset: True news articles from various sources
- Libraries: Flask, scikit-learn, pandas, NumPy
- Inspiration: Combatting misinformation in the digital age

## 📈 Future Enhancements

- [ ] Add fake news dataset for better training
- [ ] Implement confidence scores
- [ ] Add support for multiple languages
- [ ] Create API endpoints for integration
- [ ] Add user authentication and history
- [ ] Implement real-time news monitoring

## 🐛 Known Issues

- Currently trained only on true news (similarity-based approach)
- May not detect sophisticated fake news
- Requires proper news formatting for best results

## 📞 Support

If you have any questions or need help, please:
- Open an issue on GitHub
- Contact the author via email
- Check the documentation

---

**⭐ Star this repository if you found it helpful!**
