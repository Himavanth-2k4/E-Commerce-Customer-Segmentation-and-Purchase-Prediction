document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('predict-form');
  const resultEl = document.getElementById('result');
  const submitBtn = document.getElementById('submit-btn');

  function showResult(html){
    resultEl.innerHTML = html;
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    submitBtn.disabled = true;
    submitBtn.textContent = 'Predicting...';
    showResult('');

    const payload = {
      Gender: document.getElementById('Gender').value,
      Age: parseInt(document.getElementById('Age').value, 10),
      City: document.getElementById('City').value,
      Membership_Type: document.getElementById('Membership_Type').value,
      Total_Spend: parseFloat(document.getElementById('Total_Spend').value),
      Items_Purchased: parseInt(document.getElementById('Items_Purchased').value, 10),
      Average_Rating: parseFloat(document.getElementById('Average_Rating').value),
      Days_Since_Last_Purchase: parseInt(document.getElementById('Days_Since_Last_Purchase').value, 10)
    };

    try{
      const resp = await fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });

      const data = await resp.json();
      if(!resp.ok || data.status === 'error'){
        showResult(`<div class="card"><div class="prediction">Error</div><div class="muted">${data.error || JSON.stringify(data)}</div></div>`);
      } else {
        const title = `Prediction: ${data.satisfaction_level}`;
        let probs = '';
        if(data.probability_breakdown){
          const entries = Object.entries(data.probability_breakdown).map(([k,v]) => `<span class="prob-item">${k}: ${v}%</span>`).join('');
          probs = `<div class="prob-list">${entries}</div>`;
        }
        showResult(`<div class="card"><div class="prediction">${title}</div>${probs}</div>`);
      }
    } catch (err){
      showResult(`<div class="card"><div class="prediction">Network Error</div><div class="muted">${err.message}</div></div>`);
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = 'Predict';
    }
  });
});
