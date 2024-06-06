const App = () => {
    const images = [
        "https://via.placeholder.com/600x400?text=Image+1",
        "https://via.placeholder.com/600x400?text=Image+2",
        "https://via.placeholder.com/600x400?text=Image+3"
    ];

    const renderSlides = () => {
        return images.map((image, index) => (
            <div key={index} className="slide">
                <img src={image} alt={`Slide ${index + 1}`} />
            </div>
        ));
    };

    return (
        <div>
            <h1 className="heading">Welcome to Vessel Predictions</h1>
            <div className="image-container">
                <img src="https://via.placeholder.com/800x400?text=Vessel" alt="Vessel" />
            </div>
            <div className="slideshow-container">
                {renderSlides()}
            </div>
            <div className="frame">Predictive Maintenance</div>
            <div className="frame">Energy Consumption</div>
            <div className="frame">Route Optimization</div>
        </div>
    );
};

ReactDOM.render(<App />, document.getElementById('root'));