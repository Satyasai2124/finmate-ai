# AI-Assisted Personal Financial Stability Support System

## 📋 Project Overview

An intelligent financial management system that leverages machine learning to provide personalized financial insights, spending predictions, and stability assessments. The system helps users track expenses, receive real-time alerts, and make informed financial decisions.

### 🎯 Key Features
- **Smart Transaction Management**: Track income and expenses with detailed categorization
- **AI-Powered Predictions**: Machine learning models predict future spending patterns
- **Financial Stability Score**: Real-time assessment of financial health (0-100 scale)
- **Intelligent Alerts**: Automatic detection of overspending and budget violations
- **Personalized Recommendations**: Context-aware financial advice based on spending behavior
- **Category Analytics**: Detailed breakdown of expenses by category
- **Secure Authentication**: JWT-based authentication system

### 🛠️ Technology Stack
- **Backend**: FastAPI (Python)
- **Database**: SQLAlchemy with SQLite
- **ML Framework**: Scikit-learn
- **Authentication**: JWT (JSON Web Tokens)
- **Data Processing**: Pandas, NumPy

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd financial-stability-system
```

### Step 2: Install Dependencies
```bash
pip3 install -r requirements.txt
```

### Step 3: Run the Application
```bash
python3 app.py
```

### Access Points
- **Application**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Interactive API**: http://localhost:8000/redoc

## Temporary demo deployment (Render Free)

The included `render.yaml` defines a free Python web service and free PostgreSQL database.
Render's free web service can take about a minute to wake after 15 minutes without traffic.
The free PostgreSQL database is limited to 1 GB and expires 30 days after creation. After
the 14-day recovery grace period, Render deletes the database and its data. This setup is
for a temporary demo only; do not use it for real financial records or production data.

To publish the demo, connect this project to GitHub in Render and create a Blueprint from
the repository. Render will create the service, database, generated `SECRET_KEY`, and
health check configuration. The service does not seed the demo account in production.
Users must register their own accounts.

The public URL is assigned by Render after the Blueprint deploy succeeds.

For durable production hosting, use a paid web service and persistent PostgreSQL database,
or configure another trusted database provider. Review provider pricing and backup policies
before using real financial data.

For a manual deployment, configure:

- `ENVIRONMENT=production`
- `SECRET_KEY` to a unique random secret
- `DATABASE_URL` to a persistent PostgreSQL connection string
- `CORS_ORIGINS` only when a separate trusted frontend origin is required

Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## 📡 API Endpoints

### Authentication
| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/register` | Create new user account | ❌ |
| POST | `/login` | Login and receive JWT token | ❌ |

### Transaction Management
| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/transactions` | Add income/expense transaction | ✅ |
| GET | `/transactions` | Retrieve all user transactions | ✅ |

### Analytics & Insights
| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/dashboard` | Financial dashboard with stability score | ✅ |
| GET | `/alerts` | Get overspending alerts | ✅ |
| GET | `/recommendations` | Get AI-powered recommendations | ✅ |

## 💡 Usage Examples

### 1. User Registration
```bash
curl -X POST "http://localhost:8000/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "securepass123",
    "name": "John Doe",
    "monthly_income": 50000,
    "location": "Hyderabad, India"
  }'
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### 2. User Login
```bash
curl -X POST "http://localhost:8000/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=securepass123"
```

### 3. Add Transaction
```bash
curl -X POST "http://localhost:8000/transactions" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 500,
    "category": "food",
    "type": "expense",
    "description": "Grocery shopping"
  }'
```

### 4. View Dashboard
```bash
curl -X GET "http://localhost:8000/dashboard" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response:**
```json
{
  "total_income": 50000,
  "total_expenses": 15000,
  "balance": 35000,
  "stability_score": 85.5,
  "predicted_balance": 32000,
  "category_breakdown": {
    "food": 5000,
    "transport": 3000,
    "entertainment": 2000
  }
}
```

## 🤖 Machine Learning Features

### 1. Spending Prediction Model
- Uses Linear Regression to forecast next month's expenses
- Analyzes historical transaction patterns
- Provides early warnings for budget overruns

### 2. Financial Stability Score
- Calculates a 0-100 score based on income-to-expense ratio
- **100-80**: Excellent financial health
- **79-60**: Good, with room for improvement
- **59-40**: Caution advised
- **Below 40**: Critical, immediate action needed

### 3. Overspending Detection
- Real-time monitoring of category-wise spending
- Automatic alerts when thresholds are exceeded
- Severity classification (Low, Medium, High, Critical)

### 4. Recommendation Engine
- Context-aware financial advice
- Personalized based on spending patterns
- Location-specific insights

## 📊 Expense Categories

| Category | Description |
|----------|-------------|
| `food` | Groceries, dining, food delivery |
| `transport` | Fuel, public transport, cab services |
| `entertainment` | Movies, subscriptions, hobbies |
| `utilities` | Electricity, water, internet, phone |
| `healthcare` | Medical expenses, insurance |
| `education` | Courses, books, tuition |
| `shopping` | Clothing, electronics, general shopping |
| `other` | Miscellaneous expenses |

## 📁 Project Structure

```
financial-stability-system/
├── app.py                    # Application entry point
├── main.py                   # FastAPI routes and endpoints
├── models.py                 # Database models (SQLAlchemy)
├── schemas.py                # Pydantic schemas for validation
├── auth.py                   # Authentication logic (JWT)
├── database.py               # Database configuration
├── ml_engine.py              # Machine learning algorithms
├── recommendation_engine.py  # Recommendation system
├── requirements.txt          # Python dependencies
├── static/                   # Frontend files
│   └── index.html
└── README.md                 # Project documentation
```

## 🔒 Security Features

- Password hashing using bcrypt
- JWT-based stateless authentication
- Token expiration and validation
- SQL injection prevention via ORM
- CORS middleware for API security

## 🧪 Testing

Run the test suite:
```bash
python3 test_system.py
```

## 📝 Future Enhancements

- [ ] Multi-currency support
- [ ] Budget goal setting and tracking
- [ ] Export reports (PDF/Excel)
- [ ] Mobile application
- [ ] Integration with banking APIs
- [ ] Advanced ML models (LSTM for time series)
- [ ] Social features (expense splitting)

## 👥 Contributors

- **Your Name** - Developer
- **College Name** - Institution

## 📄 License

This project is developed for educational purposes.

## 📞 Support

For queries or issues, please contact: your.email@example.com
