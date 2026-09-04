export default function AlertModal({ open, reason, onCallBack, onApproveAnyway }) {
  if (!open) return null;
  return (
    <div className="modal-overlay" role="dialog" aria-modal="true">
      <div className="modal-box">
        <div className="modal-label">Pre-transaction warning</div>
        <div className="modal-reason">{reason}</div>
        <div className="modal-actions">
          <button className="action-btn" onClick={onCallBack}>
            Call back
          </button>
          <button className="action-btn secondary" onClick={onApproveAnyway}>
            Approve anyway
          </button>
        </div>
      </div>
    </div>
  );
}
