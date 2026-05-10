const overlay = document.getElementById('modal-overlay');
const openBtn = document.getElementById('open-feedback');
const closeBtn = document.getElementById('close-modal');
const form = document.querySelector('.feedback-form');
const success = document.getElementById('feedback-success');

function openModal() {
  overlay.classList.add('active');
  overlay.setAttribute('aria-hidden', 'false');
  document.body.style.overflow = 'hidden';
}

function closeModal() {
  overlay.classList.remove('active');
  overlay.setAttribute('aria-hidden', 'true');
  document.body.style.overflow = '';
}

openBtn.addEventListener('click', openModal);
closeBtn.addEventListener('click', closeModal);

overlay.addEventListener('click', (e) => {
  if (e.target === overlay) closeModal();
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeModal();
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const data = new FormData(form);
  const res = await fetch(form.action, {
    method: 'POST',
    body: data,
    headers: { 'Accept': 'application/json' }
  });
  if (res.ok) {
    form.reset();
    form.style.display = 'none';
    success.style.display = 'block';
  }
});