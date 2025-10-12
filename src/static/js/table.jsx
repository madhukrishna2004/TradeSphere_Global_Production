const DataRow = (props) => {
  const [modalOpen, setModalOpen] = useState(false);

  const highlight = (string) => {
    return renderHighlightedContent(props.filter.split(' '), String(string));
  };

  let ukTariffRate;
  let vatRate;


if (props.trade_remedy_applies && !props.cet_applies_until_trade_remedy_transition_reviews_concluded) {
  ukTariffRate = (
    <>
      <span className="govuk-table__cell--trigger" onClick={() => setModalOpen(!modalOpen)}>
        See details
      </span>
      {modalOpen && (
        <Modal handleClick={() => setModalOpen(false)}>
          <h3>A trade remedy applies to {highlight(props.commodity)} for goods arriving from specific countries</h3>
          <p>UK Global Tariff rate: {highlight(props.ukgt_duty_rate)}</p>
          <p>This will apply from 1 January 2021.</p>
          <p>
            Read more about{" "}
            <a href="https://www.gov.uk/guidance/trade-remedies-transition-policy">trade remedies</a>.
          </p>
        </Modal>
      )}
    </>
  );
} else if (props.trade_remedy_applies && props.cet_applies_until_trade_remedy_transition_reviews_concluded) {
  ukTariffRate = (
    <>
      <span className="govuk-table__cell--trigger" onClick={() => setModalOpen(!modalOpen)}>
        See details
      </span>
      {modalOpen && (
        <Modal handleClick={() => setModalOpen(false)}>
          <h3>A trade remedy applies to {highlight(props.commodity)} for goods arriving from specific countries</h3>
          <p>UK Global Tariff rate: {highlight(props.ukgt_duty_rate)}</p>
          <p>
            EU Tariff rate: {highlight(props.cet_duty_rate)} - this will continue to apply until
            transition reviews of all products in scope of this measure have been completed. The UK Global Tariff will
            then apply.
          </p>
          <p>
            Read more about{" "}
            <a href="https://www.gov.uk/guidance/trade-remedies-transition-policy">trade remedies</a>.
          </p>
        </Modal>
      )}
    </>
  );
} else if (props.suspension_applies) {
  ukTariffRate = (
    <>
      <span className="govuk-table__cell--trigger" onClick={() => setModalOpen(!modalOpen)}>
        See details
      </span>
      {modalOpen && (
        <Modal handleClick={() => setModalOpen(false)}>
          <p>UK Global Tariff rate: {highlight(props.ukgt_duty_rate)}</p>
          <p>An Autonomous Suspension applies, this will be reviewed in due course.</p>
        </Modal>
      )}
    </>
  );
} else if (props.atq_applies) {
  ukTariffRate = (
    <>
      <span className="govuk-table__cell--trigger" onClick={() => setModalOpen(!modalOpen)}>
        See details
      </span>
      {modalOpen && (
        <Modal handleClick={() => setModalOpen(false)}>
          <p>UK Global Tariff rate: {highlight(props.ukgt_duty_rate)}</p>
          <p>
            A New Autonomous Quota of 260,000 tons will apply to the following commodity codes: 1701 13 10 and 1701 14
            10 from 1 Jan 2021, for 12 months, with an in-quota rate of 0.00%.
          </p>
          <p>This will be reviewed in line with the UK’s suspensions policy in due course.</p>
        </Modal>
      )}
    </>
  );
} else if (props['Product-specific rule of origin']) {
  ukTariffRate = (
    <>
      <span className="govuk-table__cell--trigger" onClick={() => setModalOpen(!modalOpen)}>
        See details
      </span>
      {modalOpen && (
        <Modal handleClick={() => setModalOpen(false)}>
          <h3>Product-specific rule of origin applies to {highlight(props.commodity)}</h3>
          <p>UK Global Tariff rate: {highlight(props.ukgt_duty_rate)}</p>
          <p>
            This rule of origin applies specifically to the product described. Learn more from the{" "}
            <a href="https://www.gov.uk/rules-of-origin">rules of origin policy</a>.
          </p>
        </Modal>
      )}
    </>
  );
} else {
  ukTariffRate = highlight(props.ukgt_duty_rate);
}


  // VAT Rate (assuming it's coming from props['VAT'])
  vatRate = highlight(props['VAT']); // Treat 'VAT' prop in the same way as other props.

  return (
    <tr className="govuk-table__row" role="row">
      <td className="govuk-table__cell hs-cell">
        <span className="hs-cell__heading">{props.commodity.slice(0, 4)}</span>
        <span className="hs-cell__subheading">{props.commodity.slice(4, 6)}</span>
        <span className="hs-cell__subheading">{props.commodity.slice(6, 8)}</span>
        
        
      </td>
      <td className="govuk-table__cell">{highlight(props.description)}</td>
      <td className="govuk-table__cell">{highlight(props.cet_duty_rate)}</td>
      <td className="govuk-table__cell">{ukTariffRate}</td>
      <td className="govuk-table__cell">{vatRate}</td> {/* New VAT column */}
      <td className="govuk-table__cell r">{highlight(props['Product-specific rule of origin'])}</td>
    </tr>
  );
};

const DataTable = (props) => {
  return (
    <table style={{ tableLayout: 'fixed', width: '100%' }}>
      <thead>
        <tr>
          <th
            className="nw govuk-table__header govuk-table__cell sorting_asc"
            style={{ width: 104, color: "white", wordWrap: "break-word" }}
            rowSpan="1"
            colSpan="1"
            aria-label="Commodity"
          >
            Commodity
          </th>
          <th
            className="nw govuk-table__header govuk-table__cell sorting_disabled"
            style={{ width: 439, color: "white", wordWrap: "break-word" }}
            rowSpan="1"
            colSpan="1"
            aria-label="Description"
          >
            Description
          </th>
          <th
            className="nw govuk-table__header govuk-table__cell sorting_disabled"
            style={{ width: 181, color: "white", wordWrap: "break-word" }}
            rowSpan="1"
            colSpan="1"
            aria-label="EU Tariff"
          >
            EU Tariff
          </th>
          <th
            className="nw govuk-table__header govuk-table__cell sorting_disabled"
            style={{ width: 121, color: "white", wordWrap: "break-word" }}
            rowSpan="1"
            colSpan="1"
            aria-label="UK Global Tariff"
          >
            UK Global Tariff
          </th>
          <th
            className="nw govuk-table__header govuk-table__cell sorting_disabled"
            style={{ width: 121, color: "white", wordWrap: "break-word" }}
            rowSpan="1"
            colSpan="1"
            aria-label="VAT"
          >
            VAT {/* New VAT column header */}
          </th>
          <th
            className="nw govuk-table__header r govuk-table__cell sorting_disabled"
            style={{ width: 94, color: "white", wordWrap: "break-word" }}
            rowSpan="1"
            colSpan="1"
            aria-label="Product-specific rule of origin"
          >
            Product-specific rule of origin
          </th>
        </tr>
      </thead>
      <tbody>
        {/* Render rows based on props.data */}
      </tbody>
    </table>
  );
};
