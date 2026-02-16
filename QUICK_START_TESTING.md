# Quick Start Testing Guide

## 🚀 Mulai Testing dalam 5 Menit

### Prerequisite
```bash
# Sudah terinstall:
# - pytest (9.0.2)
# - pytest-cov (7.0.0)
```

### 1️⃣ Jalankan Semua Tests
```bash
python -m pytest tests/ -v
```

**Expected Output:**
```
============================= test session starts =============================
...
============================== 49 passed in 9.09s ============================
```

### 2️⃣ Jalankan Tests Specific Category

#### Configuration Tests
```bash
pytest tests/test_config.py -v
# Result: 10 tests PASSED
```

#### Data Loader Tests
```bash
pytest tests/test_data_loader.py -v
# Result: 8 tests PASSED
```

#### Model Tests
```bash
pytest tests/test_models.py -v
# Result: 8 tests PASSED
```

#### Integration Tests
```bash
pytest tests/test_integration.py -v
# Result: 23 tests PASSED
```

### 3️⃣ Jalankan dengan Coverage Report
```bash
pytest tests/ --cov=src --cov-report=term-missing
```

### 4️⃣ Jalankan Single Test
```bash
pytest tests/test_data_loader.py::TestDataLoader::test_load_data_success -v
```

### 5️⃣ Jalankan dengan Detailed Output
```bash
pytest tests/ -vv --tb=long
```

## 📊 Test Summary

| Module | Tests | Status |
|--------|-------|--------|
| integration | 23 | ✅ PASSED |
| config | 10 | ✅ PASSED |
| data_loader | 8 | ✅ PASSED |
| models | 8 | ✅ PASSED |
| **TOTAL** | **49** | **✅ PASSED** |

## 🔍 What's Tested?

### ✅ Project Structure
- Semua directories exist (src, data, models, tests)
- Semua modules can be imported
- requirements.txt has all dependencies

### ✅ Configuration
- Paths are correct
- Features properly defined
- No duplicate features

### ✅ Data Loading
- CSV file loads successfully
- Data sorted by DEPTH
- No missing depth values
- Multiple loads are consistent

### ✅ Model Manager
- Models initialize correctly
- Predictions work (porosity, fluid, pressure)
- Feature selection works
- NaN values handled properly

## 💡 Tips

### Run tests faster with parallel execution
```bash
pip install pytest-xdist
pytest tests/ -n auto
```

### Generate HTML coverage report
```bash
pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html in browser
```

### Run only failed tests
```bash
pytest tests/ --lf
```

### Run specific test pattern
```bash
pytest tests/ -k "data_loader" -v
pytest tests/ -k "not integration" -v
```

## 📝 Test Files Structure

```
tests/
├── __init__.py              # Package init
├── conftest.py              # Fixtures
├── test_integration.py       # 23 tests (structure & integration)
├── test_config.py            # 10 tests (configuration)
├── test_data_loader.py       # 8 tests (data loading)
├── test_models.py            # 8 tests (model manager)
├── test_plots.py             # (structure - can be expanded)
└── test_shap_explainer.py    # (structure - can be expanded)
```

## 🎯 Key Features Tested

1. **Data Pipeline**
   - ✅ Load CSV data
   - ✅ Validate columns
   - ✅ Sort by depth
   - ✅ Handle missing values

2. **Model Management**
   - ✅ Load ML models
   - ✅ Make predictions
   - ✅ Select features
   - ✅ Handle edge cases

3. **Configuration**
   - ✅ Path validation
   - ✅ Feature definition
   - ✅ Consistency checks

4. **Integration**
   - ✅ Project structure
   - ✅ Module imports
   - ✅ Dependency checking

## ⚡ Commands Cheat Sheet

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src

# Run specific test
pytest tests/test_config.py -v

# Run specific test class
pytest tests/test_config.py::TestConfigPaths -v

# Run specific test function
pytest tests/test_config.py::TestConfigPaths::test_root_dir_exists -v

# Run tests matching pattern
pytest tests/ -k "config" -v

# Run with more details
pytest tests/ -vv --tb=long

# Run and stop on first failure
pytest tests/ -x

# Run last failed tests
pytest tests/ --lf

# Run tests in random order
pytest tests/ --random-order

# Generate coverage HTML report
pytest tests/ --cov=src --cov-report=html
```

## 🔧 If Tests Fail

### Check Python Version
```bash
python --version
# Expected: Python 3.12.12
```

### Check Pytest Installation
```bash
pytest --version
# Expected: pytest 9.0.2
```

### Check Dependencies
```bash
pip list | findstr /I "pandas numpy matplotlib scikit-learn xgboost"
```

### Run with Full Traceback
```bash
pytest tests/ --tb=long -vv
```

## 📚 Documentation Files

- **TEST_REPORT.md** - Detailed test execution report
- **TESTING_GUIDE.md** - Comprehensive testing documentation
- **pytest.ini** - Pytest configuration

## ✨ Next Steps

1. Review TEST_REPORT.md for detailed results
2. Read TESTING_GUIDE.md for comprehensive documentation
3. Expand tests for plots and SHAP explainer if needed
4. Setup CI/CD with GitHub Actions

---

**Status:** ✅ All tests passing
**Date:** February 12, 2026
**Duration:** ~9 seconds
