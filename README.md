# Ship and Maritime Monitoring System

## Overview
The Ship and Maritime Monitoring System is designed to provide efficient tracking and management of maritime vessels. This system aims to enhance safety, improve operational efficiency, and ensure regulatory compliance in maritime activities.

## Features
- Real-time tracking of ships
- Automated alerting system for safety issues
- Data analytics for operational improvements
- Compliance monitoring for maritime regulations

## Technologies Used
- GPS and AIS for tracking
- Cloud computing for data storage and processing
- Web-based interface for user interaction

## Getting Started

### 1) Install dependencies
```bash
pip install streamlit pillow opencv-python numpy
```

### 2) Run the Streamlit app
```bash
streamlit run app.py
```

### 3) Use the application
- Upload a SAR image file (PNG/JPG/JPEG/BMP/TIF/TIFF)
- View the original image and annotated detection output
- Review dashboard metrics for traffic, security status, and trade analysis

> The current app includes a mock CNN pipeline and clearly marked `TODO` comments for integrating your trained model.

## Contributing
Contributions are welcome! Please read the CONTRIBUTING.md file for details on our code of conduct and the process for submitting pull requests to us.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
