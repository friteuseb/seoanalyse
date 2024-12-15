function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.style.backgroundColor = '#ff5252';
    errorDiv.style.color = 'white';
    errorDiv.style.padding = '10px';
    errorDiv.style.margin = '10px';
    errorDiv.style.borderRadius = '4px';
    errorDiv.textContent = message;
    
    const container = document.querySelector('#controls');
    container.insertBefore(errorDiv, container.firstChild);
    
    setTimeout(() => errorDiv.remove(), 5000);
}

