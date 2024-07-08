// crawl_dashboard.js

document.addEventListener('DOMContentLoaded', function () {
    fetchCrawls();
});

function fetchCrawls() {
    axios.get('api.php?action=getCrawls')
        .then(response => {
            const crawls = response.data;
            displayCrawls(crawls);
            displayKPIs(crawls);
            displayOverviewChart(crawls);
            displayClusterDistribution(crawls);
            displayTopNodes(crawls);
        })
        .catch(error => {
            console.error('Error fetching crawls:', error);
        });
}

function fetchCrawlDetails(crawlId) {
    axios.get(`api.php?action=getCrawlDetails&id=${crawlId}`)
        .then(response => {
            displayCrawlDetails(response.data);
        })
        .catch(error => {
            console.error('Error fetching crawl details:', error);
        });
}

function displayCrawls(crawls) {
    const crawlList = document.getElementById('crawl-list');
    crawlList.innerHTML = '<h2>Liste des Crawls</h2>';
    const ul = document.createElement('ul');

    crawls.forEach(crawl => {
        const li = document.createElement('li');
        li.textContent = `ID: ${crawl.id}, Documents: ${crawl.count}`;
        li.addEventListener('click', () => fetchCrawlDetails(crawl.id));
        ul.appendChild(li);
    });

    crawlList.appendChild(ul);
}

function displayCrawlDetails(details) {
    const crawlDetails = document.getElementById('crawl-details');
    crawlDetails.innerHTML = '<h2>Détails du Crawl</h2>';
    
    const table = document.createElement('table');
    const thead = document.createElement('thead');
    const tbody = document.createElement('tbody');

    thead.innerHTML = `
        <tr>
            <th>ID</th>
            <th>URL</th>
            <th>Label</th>
            <th>Cluster</th>
            <th>Liens Internes</th>
        </tr>
    `;

    details.forEach(doc => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${doc.doc_id}</td>
            <td>${doc.url}</td>
            <td>${doc.label}</td>
            <td>${doc.cluster}</td>
            <td>${doc.internal_links_out.length}</td>
        `;
        tbody.appendChild(tr);
    });

    table.appendChild(thead);
    table.appendChild(tbody);
    crawlDetails.appendChild(table);

    const linksData = details.map(doc => ({
        url: doc.url,
        count: doc.internal_links_out.length
    }));

    displayLinksChart(linksData);
}

function displayLinksChart(data) {
    const canvas = document.createElement('canvas');
    document.getElementById('crawl-details').appendChild(canvas);

    new Chart(canvas, {
        type: 'bar',
        data: {
            labels: data.map(d => d.url),
            datasets: [{
                label: 'Nombre de liens internes',
                data: data.map(d => d.count),
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

function displayKPIs(crawls) {
    const kpiSection = document.getElementById('kpi-section');
    const totalDocuments = crawls.reduce((sum, crawl) => sum + parseInt(crawl.count), 0);
    const averageDocuments = (totalDocuments / crawls.length).toFixed(2);

    kpiSection.innerHTML = `
        <div class="col-md-4">
            <div class="card text-white bg-primary mb-3">
                <div class="card-body">
                    <h5 class="card-title">Total Crawls</h5>
                    <p class="card-text">${crawls.length}</p>
                </div>
            </div>
        </div>
        <div class="col-md-4">
            <div class="card text-white bg-success mb-3">
                <div class="card-body">
                    <h5 class="card-title">Total Documents</h5>
                    <p class="card-text">${totalDocuments}</p>
                </div>
            </div>
        </div>
        <div class="col-md-4">
            <div class="card text-white bg-warning mb-3">
                <div class="card-body">
                    <h5 class="card-title">Average Documents per Crawl</h5>
                    <p class="card-text">${averageDocuments}</p>
                </div>
            </div>
        </div>
    `;
}

function displayOverviewChart(crawls) {
    const overviewContent = document.getElementById('overview-content');
    overviewContent.innerHTML = '';  // Clear previous content

    const canvas = document.createElement('canvas');
    overviewContent.appendChild(canvas);

    const crawlLabels = crawls.map(crawl => crawl.id);
    const documentCounts = crawls.map(crawl => parseInt(crawl.count));

    new Chart(canvas, {
        type: 'bar',
        data: {
            labels: crawlLabels,
            datasets: [{
                label: 'Nombre de documents par crawl',
                data: documentCounts,
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

function displayClusterDistribution(crawls) {
    const clustersContent = document.getElementById('clusters-content');
    clustersContent.innerHTML = '';  // Clear previous content

    const clusterCounts = {};
    crawls.forEach(crawl => {
        if (crawl.documents) {
            crawl.documents.forEach(doc => {
                if (clusterCounts[doc.cluster]) {
                    clusterCounts[doc.cluster]++;
                } else {
                    clusterCounts[doc.cluster] = 1;
                }
            });
        }
    });

    const canvas = document.createElement('canvas');
    clustersContent.appendChild(canvas);

    new Chart(canvas, {
        type: 'pie',
        data: {
            labels: Object.keys(clusterCounts),
            datasets: [{
                data: Object.values(clusterCounts),
                backgroundColor: ['#ff6384', '#36a2eb', '#cc65fe', '#ffce56', '#ff6384'],
                hoverBackgroundColor: ['#ff6384', '#36a2eb', '#cc65fe', '#ffce56', '#ff6384']
            }]
        },
        options: {
            responsive: true
        }
    });
}

function displayTopNodes(crawls) {
    const topNodesContent = document.getElementById('top-nodes-content');
    topNodesContent.innerHTML = '';  // Clear previous content

    const nodeLinks = {};
    crawls.forEach(crawl => {
        if (crawl.documents) {
            crawl.documents.forEach(doc => {
                const internalLinksCount = doc.internal_links_out.length;
                if (nodeLinks[doc.url]) {
                    nodeLinks[doc.url] += internalLinksCount;
                } else {
                    nodeLinks[doc.url] = internalLinksCount;
                }
            });
        }
    });

    const sortedNodes = Object.entries(nodeLinks).sort((a, b) => b[1] - a[1]);
    const topNodes = sortedNodes.slice(0, 10);  // Get top 10 nodes

    const canvas = document.createElement('canvas');
    topNodesContent.appendChild(canvas);

    new Chart(canvas, {
        type: 'bar',
        data: {
            labels: topNodes.map(node => node[0]),
            datasets: [{
                label: 'Nombre de liens internes',
                data: topNodes.map(node => node[1]),
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}
