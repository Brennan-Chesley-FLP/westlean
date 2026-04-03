import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  mainSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Algorithms',
      items: [
        'algorithms/overview',
        'algorithms/exalg',
        'algorithms/anti-unification',
        'algorithms/fivatech',
        'algorithms/roadrunner',
        'algorithms/k-testable',
      ],
    },
  ],
  apiSidebar: [
    'api/serialization',
    'api/tracer',
  ],
};

export default sidebars;
