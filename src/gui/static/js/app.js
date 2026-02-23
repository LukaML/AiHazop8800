// app.js - Main application coordinator

document.addEventListener('DOMContentLoaded', () => {
    // Initialize components
    SetupPanel.init();
    ResultsTable.init();
    EditModal.init();
    BulkRegenModal.init();
    ChangesModal.init();

    console.log('HAZOP GUI initialized');
});
